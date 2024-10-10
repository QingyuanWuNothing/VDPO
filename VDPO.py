import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from rich import print
import numpy as np
from tqdm import trange
from tensorboardX import SummaryWriter
from collections import deque

from rich import print
from collections import deque
from tqdm import tqdm, trange
from make_env import make_mujoco_env, make_delayed_mujoco_env
from belief import TransBelief
from nn import SAC_Actor, SAC_Critic
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", type=str, default=os.path.basename(__file__).rstrip(".py"))
parser.add_argument("--env", type=str, default="Ant-v4")
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--device", default=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--buffer_size", type=int, default=int(1e6))
parser.add_argument("--total_timesteps", type=int, default=int(1e6))
parser.add_argument("--learn_start", type=int, default=int(5e3))
parser.add_argument("--batch_size", type=int, default=int(256))
parser.add_argument("--actor_lr", type=float, default=3e-4)
parser.add_argument("--critic_lr", type=float, default=1e-3)
parser.add_argument("--alpha_lr", type=float, default=1e-3)
parser.add_argument("--target_update_factor", type=float, default=5e-3)
parser.add_argument("--actor_freq", type=int, default=int(2))
parser.add_argument("--target_freq", type=int, default=int(1))
parser.add_argument("--eval_freq", type=int, default=int(1e4))
parser.add_argument("--eval_num", type=int, default=int(1e1))
parser.add_argument("--kl_freq", type=int, default=int(1e3))
parser.add_argument("--belief_lr", type=float, default=3e-4)
parser.add_argument("--embedding_dim", type=int, default=int(256))
parser.add_argument("--n_steps", type=int, default=int(3))
parser.add_argument("--delay", type=int, default=int(5))


def kl_divergence(mu1, sigma1, mu2, sigma2):
    kl = torch.log(sigma2 / sigma1) + (sigma1**2 + (mu1 - mu2)**2) / (2 * sigma2**2) - 0.5
    return kl

class ReplayBuffer():
    def __init__(self, buffer_size, observation_dim, action_dim, seq_len):
        super().__init__()
        self.buffer = {
            # sac
            'obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'actions': torch.zeros((buffer_size, action_dim), dtype=torch.float32),
            'rewards': torch.zeros(buffer_size, 1, dtype=torch.float32),
            'nxt_obs': torch.zeros((buffer_size, observation_dim), dtype=torch.float32),
            'dones': torch.zeros(buffer_size, 1, dtype=torch.float32),
            'discount_factors': torch.zeros(buffer_size, 1, dtype=torch.float32),

            # belief
            'belief_states': torch.zeros((buffer_size, 1, observation_dim), dtype=torch.float32),
            'belief_actions': torch.zeros((buffer_size, seq_len, action_dim), dtype=torch.float32),
            'belief_time_steps': torch.zeros((buffer_size, seq_len), dtype=torch.int64),
            'belief_padding_masks': torch.zeros((buffer_size, seq_len), dtype=torch.bool),
            'belief_target_states': torch.zeros((buffer_size, seq_len, observation_dim), dtype=torch.float32),
        }

        # for key in self.buffer.keys():
        #     print(f"{key}: {self.buffer[key].shape}")

        self.buffer_size = buffer_size
        self.buffer_len = 0
        self.buffer_ptr = 0

    def store(self, obs, action, reward, nxt_obs, done, discount_factor,
                    belief_states, belief_actions, belief_time_steps, belief_padding_masks, belief_target_states):

        self.buffer['obs'][self.buffer_ptr] = obs.squeeze(0)
        self.buffer['actions'][self.buffer_ptr] = action.squeeze(0)
        self.buffer['rewards'][self.buffer_ptr] = torch.tensor(reward, dtype=torch.float32).unsqueeze(0)
        self.buffer['nxt_obs'][self.buffer_ptr] = nxt_obs.squeeze(0)
        self.buffer['dones'][self.buffer_ptr] = torch.tensor(done, dtype=torch.float32).unsqueeze(0)
        self.buffer['discount_factors'][self.buffer_ptr] = torch.FloatTensor(discount_factor)
        
        self.buffer['belief_states'][self.buffer_ptr] = belief_states.squeeze(0)
        self.buffer['belief_actions'][self.buffer_ptr] = belief_actions.squeeze(0)
        self.buffer['belief_time_steps'][self.buffer_ptr] = belief_time_steps.squeeze(0)
        self.buffer['belief_padding_masks'][self.buffer_ptr] = belief_padding_masks.squeeze(0)
        self.buffer['belief_target_states'][self.buffer_ptr] = belief_target_states.squeeze(0)

        self.buffer_ptr += 1
        if self.buffer_ptr >= self.buffer_size:
            self.buffer_ptr = 0

        if self.buffer_len < self.buffer_size:
            self.buffer_len += 1

    def sample(self, batch_size, device):
        indices = np.random.choice(self.buffer_len, size=batch_size)
        b_obs = self.buffer['obs'][indices].to(device)
        b_actions = self.buffer['actions'][indices].to(device)
        b_rewards = self.buffer['rewards'][indices].to(device)
        b_nxt_obs = self.buffer['nxt_obs'][indices].to(device)
        b_dones = self.buffer['dones'][indices].to(device)
        b_discount_factors = self.buffer['discount_factors'][indices].to(device)

        b_belief_states = self.buffer['belief_states'][indices].to(device)
        b_belief_actions = self.buffer['belief_actions'][indices].to(device)
        b_belief_time_steps = self.buffer['belief_time_steps'][indices].to(device)
        b_belief_padding_masks = self.buffer['belief_padding_masks'][indices].to(device)
        b_belief_target_states = self.buffer['belief_target_states'][indices].to(device)
        return [b_obs, b_actions, b_rewards, b_nxt_obs, b_dones, b_discount_factors, b_belief_states, b_belief_actions, b_belief_time_steps, b_belief_padding_masks, b_belief_target_states]

class Trainer:
    def __init__(self, config, exp_tag):
        ###############################################################################
        ################################## SETUP ######################################
        ###############################################################################
        self.config = config
        self.exp_tag = exp_tag
        self.logger = SummaryWriter(self.exp_tag)
        self.logger.add_text(
            "config",
            "|parametrix|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in config.items()])),
        )
        self.log_dict = {}
        print('✓ setup')
        ###############################################################################
        ################################### ENV #######################################
        ###############################################################################

        self.env = make_mujoco_env(env_name=config["env"], seed=config["seed"])
        self.eval_env = make_delayed_mujoco_env(env_name=config["env"], seed=config["seed"], delay=config["delay"])
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_high = float(self.env.action_space.high[0])
        self.action_low = float(self.env.action_space.low[0])

        ###############################################################################
        ################################## BUFFER #####################################
        ###############################################################################

        self.replay_buffer = ReplayBuffer(config["buffer_size"], self.observation_dim, self.action_dim, config["delay"])

        ###############################################################################
        ############################## Delayed Policy #################################
        ###############################################################################

        self.belief = TransBelief(
            observation_dim=self.observation_dim, 
            action_dim=self.action_dim,            
            action_high=self.action_high, 
            action_low=self.action_low,
            logstd_min=-5, 
            logstd_max=2,
            seq_len=config["delay"],
            embedding_dim=config["embedding_dim"],
            num_layers=3,
            num_heads=1,
            attention_dropout=0.1,
            residual_dropout=0.1,
            embedding_dropout=0.1,
        ).to(config["device"])
        self.belief_optimizer = torch.optim.Adam(
            self.belief.parameters(), 
            lr=config["belief_lr"],
        )

        ###############################################################################
        ############################# Reference Policy ################################
        ###############################################################################

        self.actor = SAC_Actor(
            observation_dim=self.observation_dim, 
            action_dim=self.action_dim, 
            action_high=self.action_high, 
            action_low=self.action_low).to(config["device"])
        self.actor_optimizer = optim.Adam(list(self.actor.parameters()), lr=config["actor_lr"])

        self.critic_1 = SAC_Critic(self.observation_dim, self.action_dim).to(config["device"])
        self.target_1 = SAC_Critic(self.observation_dim, self.action_dim).to(config["device"])
        self.target_1.load_state_dict(self.critic_1.state_dict())
        self.target_1.eval()

        self.critic_2 = SAC_Critic(self.observation_dim, self.action_dim).to(config["device"])
        self.target_2 = SAC_Critic(self.observation_dim, self.action_dim).to(config["device"])
        self.target_2.load_state_dict(self.critic_2.state_dict())
        self.target_2.eval()

        self.critic_optimizer = optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=config["critic_lr"])

        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(config["device"])).item()

        self.log_alpha = torch.zeros(1, requires_grad=True, device=config["device"])
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config["alpha_lr"])

    def train_sac(self, batch):
        b_obs, b_actions, b_rewards, b_nxt_obs, b_dones, b_discount_factors = batch
        # update critic
        with torch.no_grad():
            nxt_obs_actions, nxt_obs_log_pi, _ = self.actor.get_action(b_nxt_obs)
            target_1_nxt_value = self.target_1(b_nxt_obs, nxt_obs_actions)
            target_2_nxt_value = self.target_2(b_nxt_obs, nxt_obs_actions)
            min_target_nxt_value = torch.min(target_1_nxt_value, target_2_nxt_value) - self.alpha * nxt_obs_log_pi
            nxt_q_values = b_rewards.flatten() + (1 - b_dones.flatten()) * b_discount_factors.flatten() * (min_target_nxt_value).view(-1)

        critic_1_values = self.critic_1(b_obs, b_actions).view(-1)
        critic_2_values = self.critic_2(b_obs, b_actions).view(-1)
        critic_1_loss = F.mse_loss(critic_1_values, nxt_q_values)
        critic_2_loss = F.mse_loss(critic_2_values, nxt_q_values)
        critic_loss = critic_1_loss + critic_2_loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.log_dict["train/critic_loss"] = critic_loss.item()
        self.log_dict["train/critic_value"] = critic_1_values.mean().item()

        if self.global_step % self.config["actor_freq"]:
            for _ in range(self.config["actor_freq"]):
                # update actor
                actions, log_prob, _ = self.actor.get_action(b_obs)
                critic_1_q_value = self.critic_1(b_obs, actions)
                critic_2_q_value = self.critic_2(b_obs, actions)
                min_critic_q_value = torch.min(critic_1_q_value, critic_2_q_value)
                actor_loss = ((self.alpha * log_prob) - min_critic_q_value).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.log_dict["train/actor_loss"] = actor_loss.item()

                # update alpha
                with torch.no_grad():
                    _, log_prob, _ = self.actor.get_action(b_obs)
                alpha_loss = (-self.log_alpha.exp() * (log_prob + self.target_entropy)).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
                self.log_dict["train/alpha_loss"] = alpha_loss.item()

    def train_belief(self, batch):

        b_belief_states, b_belief_actions, b_belief_time_steps, b_belief_padding_masks, b_belief_target_states = batch
        masks = ~b_belief_padding_masks

        trans_rec = self.belief.get_rec_state(
            observations = b_belief_states, 
            actions = b_belief_actions,
            time_steps = b_belief_time_steps,
            padding_masks = b_belief_padding_masks,
        )

        belief_loss = F.mse_loss(trans_rec, b_belief_target_states, reduction="none").mean(-1)
        belief_loss = (masks * belief_loss).mean()

        self.belief_optimizer.zero_grad()
        belief_loss.backward()
        self.belief_optimizer.step()        
        self.log_dict["train/belief_loss"] = belief_loss.item()

    def train_kl(self, batch):
        b_belief_states, b_belief_actions, b_belief_time_steps, b_belief_padding_masks, b_belief_target_states = batch
        masks = ~b_belief_padding_masks
        with torch.no_grad():
            ref_mean, ref_std = self.actor.get_mean_std(b_belief_target_states)
            trans_emb = self.belief(
                observations = b_belief_states, 
                actions = b_belief_actions,
                time_steps = b_belief_time_steps,
                padding_masks = b_belief_padding_masks,
            )
        mean, std = self.belief.trans_emb_2_mean_std(trans_emb)
        self.belief_optimizer.zero_grad()
        kl_loss = kl_divergence(mean, std, ref_mean, ref_std).mean(-1)
        kl_loss = (masks * kl_loss).mean()
        kl_loss.backward()
        self.belief_optimizer.step()
        self.log_dict["train/kl_loss"] = kl_loss.item()

    def train(self):
        obs, _ = self.env.reset(seed=self.config["seed"])
        done, ep_re, ep_len = False, 0.0, 0
        obs_delay_buffer = deque(maxlen=self.config["delay"] + 1)
        obs = torch.tensor(obs).float().unsqueeze(0).to(self.config["device"])
        obs_delay_buffer.append(obs)
        act_delay_buffer = deque(maxlen=self.config["delay"])
        time_steps = torch.tensor(np.arange(0, self.config["delay"])).unsqueeze(0).to(self.config["device"])

        padding_obs = torch.zeros((1, self.config["delay"], self.observation_dim)).to(self.config["device"])
        padding_actions = torch.zeros((1, self.config["delay"], self.action_dim)).to(self.config["device"])

        rollout_n_step_buffer = {
            'obs': deque(maxlen=self.config["n_steps"] + 1),
            'action': deque(maxlen=self.config["n_steps"]),
            'reward': deque(maxlen=self.config["n_steps"]),
            'done': deque(maxlen=self.config["n_steps"]),
            'belief_state': deque(maxlen=self.config["n_steps"]),
            'belief_action': deque(maxlen=self.config["n_steps"]),
            'belief_time_step': deque(maxlen=self.config["n_steps"]),
            'belief_padding_mask': deque(maxlen=self.config["n_steps"]),
            'belief_target_state': deque(maxlen=self.config["n_steps"]),
        }
        rollout_n_step_buffer['obs'].append(obs)

        global_step_bar = trange(1, self.config["total_timesteps"] + 1)
        for self.global_step in global_step_bar:
            if self.global_step <= self.config["learn_start"]:
                act = self.env.action_space.sample()
            else:
                with torch.no_grad():
                    # delay-free
                    act_delay_free, _, _ = self.actor.get_action(obs)
                    critic_1_q_delay_free = self.critic_1(obs, act_delay_free)
                    critic_2_q_delay_free = self.critic_2(obs, act_delay_free)
                    q_delay_free = torch.min(critic_1_q_delay_free, critic_2_q_delay_free)

                    # delayed
                    cur_delay = len(act_delay_buffer)
                    if cur_delay != 0:
                        masks = np.hstack(
                            [np.ones(cur_delay), np.zeros(self.config["delay"]-cur_delay)]
                        )
                        padding_masks = ~torch.tensor(masks).to(torch.bool).unsqueeze(0).to(self.config["device"])
                        belief_states = obs_delay_buffer[0]
                        belief_actions = torch.stack(list(act_delay_buffer), dim=1)
                        if cur_delay != self.config["delay"]:
                            belief_actions = torch.concat(
                                (belief_actions, 
                                padding_actions[:, :self.config["delay"] - cur_delay, :]),
                                dim=1
                            )
                        belief_time_steps = time_steps
                        belief_padding_masks = padding_masks
                        act, _, _ = self.belief.get_action(
                            observations = belief_states, 
                            actions = belief_actions,
                            time_steps = belief_time_steps,
                            padding_masks = belief_padding_masks,
                        )
                        act = act[:, cur_delay-1, :]
                        act_delayed = act
                    else:
                        act_delayed = self.env.action_space.sample()
                        act_delayed = torch.tensor(act_delayed).float().unsqueeze(0).to(self.config["device"])

                    critic_1_q_delayed = self.critic_1(obs, act_delayed)
                    critic_2_q_delayed = self.critic_2(obs, act_delayed)
                    q_delayed = torch.min(critic_1_q_delayed, critic_2_q_delayed)

                    if q_delay_free < q_delayed:
                        act = act_delay_free.squeeze().cpu().numpy()
                    else:
                        act = act_delayed.squeeze().cpu().numpy()


                        
            nxt_obs, reward, terminated, truncated, info = self.env.step(act)
            done = np.logical_or(terminated, truncated)
            ep_re += reward
            ep_len += 1

            nxt_obs = torch.tensor(nxt_obs).float().unsqueeze(0).to(self.config["device"])
            obs_delay_buffer.append(nxt_obs)
            act = torch.tensor(act).float().unsqueeze(0).to(self.config["device"])
            act_delay_buffer.append(act)

            ################################################################
            cur_delay = len(act_delay_buffer)
            masks = np.hstack(
                [np.ones(cur_delay), np.zeros(self.config["delay"]-cur_delay)]
            )
            padding_masks = ~torch.tensor(masks).to(torch.bool).unsqueeze(0).to(self.config["device"])
            belief_states = obs_delay_buffer[0]
            belief_actions = torch.stack(list(act_delay_buffer), dim=1)
            belief_target_states = torch.stack(list(obs_delay_buffer)[1:], dim=1)
            if cur_delay != self.config["delay"]:
                belief_actions = torch.concat(
                    (belief_actions, 
                    padding_actions[:, :self.config["delay"] - cur_delay, :]),
                    dim=1
                )
                
                belief_target_states  = torch.concat(
                    (belief_target_states, 
                    padding_obs[:, :self.config["delay"] - cur_delay, :]),
                    dim=1
                )
            belief_time_steps = time_steps
            belief_padding_masks = padding_masks
            ################################################################

            obs = nxt_obs

            rollout_n_step_buffer['obs'].append(obs)
            rollout_n_step_buffer['action'].append(act)
            rollout_n_step_buffer['reward'].append(reward)
            rollout_n_step_buffer['done'].append(done)
            rollout_n_step_buffer['belief_state'].append(belief_states)
            rollout_n_step_buffer['belief_action'].append(belief_actions)
            rollout_n_step_buffer['belief_time_step'].append(belief_time_steps)
            rollout_n_step_buffer['belief_padding_mask'].append(belief_padding_masks)
            rollout_n_step_buffer['belief_target_state'].append(belief_target_states)

            if len(rollout_n_step_buffer['obs']) == self.config["n_steps"] + 1:
                n_step_returns = 0
                for i in range(self.config["n_steps"]):
                    n_step_returns += pow(self.config["gamma"], i) * rollout_n_step_buffer['reward'][i]
                    self.replay_buffer.store(
                        obs=rollout_n_step_buffer['obs'][0], 
                        action=rollout_n_step_buffer['action'][0],
                        reward=n_step_returns, 
                        nxt_obs=rollout_n_step_buffer['obs'][i+1], 
                        done=rollout_n_step_buffer['done'][i],
                        discount_factor=np.array([pow(self.config["gamma"], i+1)]),
                        belief_states=rollout_n_step_buffer['belief_state'][i],
                        belief_actions=rollout_n_step_buffer['belief_action'][i],
                        belief_time_steps=rollout_n_step_buffer['belief_time_step'][i],
                        belief_padding_masks=rollout_n_step_buffer['belief_padding_mask'][i],
                        belief_target_states=rollout_n_step_buffer['belief_target_state'][i],
                        )

            if done:
                self.log_dict["train/train_re"] = ep_re
                obs, _ = self.env.reset(seed=self.config["seed"])
                done, ep_re, ep_len = False, 0.0, 0
                obs_delay_buffer = deque(maxlen=self.config["delay"] + 1)
                obs = torch.tensor(obs).float().unsqueeze(0).to(self.config["device"])
                obs_delay_buffer.append(obs)
                act_delay_buffer = deque(maxlen=self.config["delay"])
                time_steps = torch.tensor(np.arange(0, self.config["delay"])).unsqueeze(0).to(self.config["device"])
                padding_actions = torch.zeros((1, self.config["delay"], self.action_dim)).to(self.config["device"])
                
                rollout_n_step_buffer = {
                    'obs': deque(maxlen=self.config["n_steps"] + 1),
                    'action': deque(maxlen=self.config["n_steps"]),
                    'reward': deque(maxlen=self.config["n_steps"]),
                    'done': deque(maxlen=self.config["n_steps"]),
                    'belief_state': deque(maxlen=self.config["n_steps"]),
                    'belief_action': deque(maxlen=self.config["n_steps"]),
                    'belief_time_step': deque(maxlen=self.config["n_steps"]),
                    'belief_padding_mask': deque(maxlen=self.config["n_steps"]),
                    'belief_target_state': deque(maxlen=self.config["n_steps"]),
                }
                rollout_n_step_buffer['obs'].append(obs)


            if self.global_step >= self.config["learn_start"]:
                b_obs, b_actions, b_rewards, b_nxt_obs, b_dones, b_discount_factors, b_belief_states, b_belief_actions, b_belief_time_steps, b_belief_padding_masks, b_belief_target_states = self.replay_buffer.sample(config["batch_size"], config["device"])
                # train sac
                train_batch = [b_obs, b_actions, b_rewards, b_nxt_obs, b_dones, b_discount_factors]
                self.train_sac(train_batch)
                train_batch = [b_belief_states, b_belief_actions, b_belief_time_steps, b_belief_padding_masks, b_belief_target_states]
                self.train_belief(train_batch)

                if self.global_step % self.config["kl_freq"] == 0:
                    for _ in range(self.config["kl_freq"]):
                        b_obs, b_actions, b_rewards, b_nxt_obs, b_dones, b_discount_factors, b_belief_states, b_belief_actions, b_belief_time_steps, b_belief_padding_masks, b_belief_target_states = self.replay_buffer.sample(config["batch_size"], config["device"])
                        train_batch = [b_belief_states, b_belief_actions, b_belief_time_steps, b_belief_padding_masks, b_belief_target_states]
                        self.train_kl(train_batch)

                if self.global_step % self.config["target_freq"] == 0:
                    for param, target_param in zip(self.critic_1.parameters(), self.target_1.parameters()):
                        target_param.data.copy_(self.config["target_update_factor"] * param.data + (1 - self.config["target_update_factor"]) * target_param.data)
                    for param, target_param in zip(self.critic_2.parameters(), self.target_2.parameters()):
                        target_param.data.copy_(self.config["target_update_factor"] * param.data + (1 - self.config["target_update_factor"]) * target_param.data)

            if self.global_step % self.config["eval_freq"] == 0:
                self.actor.eval()
                self.belief.eval()
                eval_re = []
                
                for _ in range(self.config["eval_num"]):
                    eval_re.append(self.rollout_trans_decision())
                print(f'global step {self.global_step}, trans_decision ep_re {np.mean(eval_re)}')
                self.log_dict["eval/trans_decision"] = np.mean(eval_re)

                self.actor.train()
                self.belief.train()
            self.logging()

    def rollout_trans_decision(self):
        obs, _ = self.eval_env.reset(seed=self.config["seed"])
        done, ep_re, ep_len = False, 0.0, 0
        obs = torch.tensor(obs).float().unsqueeze(0).to(self.config["device"])
        act_delay_buffer = deque(maxlen=self.config["delay"])
        time_steps = torch.tensor(np.arange(0, self.config["delay"])).unsqueeze(0).to(self.config["device"])
        padding_actions = torch.zeros((1, self.config["delay"], self.action_dim)).to(self.config["device"])

        while not done:                
            ################################################################
            with torch.no_grad():
                cur_delay = len(act_delay_buffer)
                if cur_delay != 0:
                    masks = np.hstack(
                        [np.ones(cur_delay), np.zeros(self.config["delay"]-cur_delay)]
                    )
                    padding_masks = ~torch.tensor(masks).to(torch.bool).unsqueeze(0).to(self.config["device"])
                    belief_states = obs
                    belief_actions = torch.stack(list(act_delay_buffer), dim=1)
                    if cur_delay != self.config["delay"]:
                        belief_actions = torch.concat(
                            (belief_actions, 
                            padding_actions[:, :self.config["delay"] - cur_delay, :]),
                            dim=1
                        )
                    belief_time_steps = time_steps
                    belief_padding_masks = padding_masks
                    _, _, act = self.belief.get_action(
                        observations = belief_states, 
                        actions = belief_actions,
                        time_steps = belief_time_steps,
                        padding_masks = belief_padding_masks,
                    )
                    act = act[:, cur_delay-1, :]
                    act = act.squeeze().cpu().numpy()
                else:
                    act = self.eval_env.action_space.sample()
            nxt_obs, reward, terminated, truncated, info = self.eval_env.step(act)

            done = np.logical_or(terminated, truncated)
            ep_re += reward
            ep_len += 1
            nxt_obs = torch.tensor(nxt_obs).float().unsqueeze(0).to(self.config["device"])
            act = torch.tensor(act).float().unsqueeze(0).to(self.config["device"])
            act_delay_buffer.append(act)
            obs = nxt_obs
        return ep_re

    def logging(self):
        for k in self.log_dict.keys():
            self.logger.add_scalar(k, self.log_dict[k], global_step=self.global_step)
        self.log_dict = {}

if __name__ == "__main__":
    config = vars(parser.parse_args())
    print(config)
    exp_tag = f'logs/{config["exp_name"]}/ENV_{config["env"]}_DELAYS_{config["delay"]}_SEED_{config["seed"]}'
    trainer = Trainer(config, exp_tag)
    trainer.train()
    