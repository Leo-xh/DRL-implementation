import gym
import math
import numpy as np
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from PrioritizedReplayMemory import PrioritizedExperienceReplayMemory
from WrapPytorch import WrapPytorch
from DQN import DQNAgent, Config


class DQNAgent_Prioritized(DQNAgent):
    def __init__(self, env, log_dir='./dqn/1'):
        return super().__init__(env, log_dir=log_dir)
    
    def declare_memory(self):
        self.memory = PrioritizedExperienceReplayMemory(self.experience_replay_size)

    def get_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, batch_next_state, indices, weights = batch_vars

        q_values = self.model(batch_state).gather(1, batch_action)
        target_q_values = batch_reward + self.gamma * \
            self.target_model(batch_next_state).detach().max(1)[0]

        loss = self.loss_func(q_values, target_q_values)
        
        diff = target_q_values - q_values
        self.memory.update_priorities(indices, diff.detach().sequeeze().abs().cpu().numpy().tolist())

        return loss


if __name__ == "__main__":
    log_dir = "./dqn-p/1"
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    env_id = "Pong-v0"
    env = gym.make(env_id)
    # env = gym.wrappers.Monitor(env, osp.join(log_dir,"record"))
    env = WrapPytorch(env)

    agent = DQNAgent(env, log_dir=log_dir)
    # agent.load(log_dir)
    episode_rewards = []

    ep = 0
    obs = env.reset()
    episode_reward = 0
    for frame in range(Config.MAX_FRAMES):
        # print("frame", frame)
        # env.render()
        epsilon = Config.epsilon_by_frame(frame)
        action = agent.get_action(obs, epsilon)
        prev_obs = obs
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        agent.update(prev_obs, action, reward, obs, frame)
        if done:
            episode_rewards.append(episode_reward)
            agent.writer.add_scalar("data/reward", episode_reward, ep)
            print("episode", ep, "reward:", episode_reward)
            ep += 1
            obs = env.reset()
            episode_reward = 0
        if ep % 50 == 0:
            agent.save(log_dir)

    agent.save(log_dir)
    env.close()
    agent.writer.close()
