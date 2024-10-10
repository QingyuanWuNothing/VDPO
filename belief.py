import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Deque
from utils import *
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from rich import print
from collections import deque
from tqdm import tqdm, trange

class TransformerBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        embedding_dim,
        num_heads,
        attention_dropout,
        residual_dropout,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = nn.MultiheadAttention(
            embedding_dim, num_heads, attention_dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Dropout(residual_dropout),
        )
        self.register_buffer(
            "causal_mask", ~torch.tril(torch.ones(seq_len, seq_len)).to(bool)
        )
        self.seq_len = seq_len

    def forward(self, x, padding_mask):
        causal_mask = self.causal_mask[: x.shape[1], : x.shape[1]]
        x = self.norm1(x)
        attention_out = self.attention(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )[0]
        # x = x + attention_out
        x = x + self.drop(attention_out)
        # x = x + self.mlp(x)
        x = x + self.mlp(self.norm2(x))
        return x



class TransBelief(nn.Module):
    def __init__(
        self,
        observation_dim,
        action_dim,
        action_high, 
        action_low,
        logstd_min, 
        logstd_max,
        seq_len,
        embedding_dim,
        num_layers,
        num_heads,
        attention_dropout,
        residual_dropout,
        embedding_dropout,
    ):
        super().__init__()

        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim

        self.observation_emb = nn.Linear(observation_dim, embedding_dim)
        self.action_emb = nn.Linear(action_dim, embedding_dim)
        self.timestep_emb = nn.Embedding(seq_len, embedding_dim)
        self.observation_action_emb = nn.Sequential(
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        

        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.out_norm = nn.LayerNorm(embedding_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    seq_len=seq_len,
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, observation_dim), 
        )

        self.actor_public = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(embedding_dim, action_dim),
        )
        self.actor_logstd = nn.Sequential(
            nn.Linear(embedding_dim, action_dim),
            nn.Tanh(),
        )
        self.register_buffer(
            "action_scale", torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32)
        )
        self.logstd_min = logstd_min
        self.logstd_max = logstd_max

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, observations, actions, time_steps, padding_masks):

        batch_size, seq_len = actions.shape[0], actions.shape[1]

        time_emb = self.timestep_emb(time_steps)

        observations = observations.repeat(1, seq_len, 1)
        observations_emb = self.observation_emb(observations)
        actions_emb = self.action_emb(actions)
        observations_actions = torch.concat((observations_emb, actions_emb), dim=-1)
        observations_actions_emb = self.observation_action_emb(observations_actions) + time_emb

        sequence = (
            torch.stack([observations_actions_emb], dim=1)
            .permute(0, 2, 1, 3)
            .reshape(batch_size,self.seq_len, self.embedding_dim)
        )
        if padding_masks is not None:
            padding_masks = (
                torch.stack([padding_masks], dim=1)
                .permute(0, 2, 1)
                .reshape(batch_size, self.seq_len)
            )

        out = self.emb_norm(sequence)
        out = self.emb_drop(out)
        # out = sequence
        for block in self.blocks:
            out = block(out, padding_mask=padding_masks)
        trans_emb = self.out_norm(out)
        return trans_emb

    def get_rec_state(self, observations, actions, time_steps, padding_masks):
        trans_emb = self(observations, actions, time_steps, padding_masks)
        trans_rec = self.decoder(trans_emb)
        return trans_rec

    def get_mean_std(self, observations, actions, time_steps, padding_masks):
        trans_emb = self(observations, actions, time_steps, padding_masks)
        public_x = self.actor_public(trans_emb)
        mean = self.actor_mean(public_x)
        logstd = self.actor_logstd(public_x)
        logstd = self.logstd_min + 0.5 * (self.logstd_max - self.logstd_min) * (logstd + 1)
        std = logstd.exp()
        return mean, std

    def trans_emb_2_mean_std(self, trans_emb):
        public_x = self.actor_public(trans_emb)
        mean = self.actor_mean(public_x)
        logstd = self.actor_logstd(public_x)
        logstd = self.logstd_min + 0.5 * (self.logstd_max - self.logstd_min) * (logstd + 1)
        std = logstd.exp()
        return mean, std

    def get_action(self, observations, actions, time_steps, padding_masks):
        mean, std = self.get_mean_std(observations, actions, time_steps, padding_masks)
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean