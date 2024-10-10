import gymnasium as gym
import numpy as np
from rich import print
from collections import deque

def make_mujoco_env(env_name, seed):
    env = gym.make(env_name)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env

class ConstantObservationDelay(gym.Wrapper):
    def __init__(self, env, delay=5):
        super().__init__(env)
        self.delay = delay
        self.obs_delay_buffer = deque(maxlen=delay+1)
        self.obs_delay = 0
    
    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.obs_delay_buffer = deque(maxlen=self.delay+1)
        self.obs_delay_buffer.append(observation)
        self.obs_delay = 0
        info['delays'] = self.obs_delay
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self.obs_delay_buffer.append(observation)
        observation = self.obs_delay_buffer[0]
        if self.obs_delay < self.delay:
            self.obs_delay += 1
        info['delays'] = self.obs_delay

        return (
            observation, 
            reward, 
            terminated, 
            truncated, 
            info
        )

def make_delayed_mujoco_env(env_name, seed, delay):
    env = gym.make(env_name)
    env = ConstantObservationDelay(env, delay)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env