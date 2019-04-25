from WrapPytorch import WrapPytorch

import sys
import math
import argparse
import gym
import os
import numpy as np
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.distributions import Categorical
from collections import namedtuple
from itertools import count


class Config(object):
    num_agents = 16
    rollout = 5
    GAMMA = 0.99
    LR = 7e-4
    entropy_loss_weight = 0.01
    value_loss_weight = 0.5
    MAX_FRAMES = 100000
    TARGET_NET_UPDATE_FREQ = 1000
    EXP_REPLAY_SIZE = 10000
    BATCH_SIZE = 32

    FEATURE_SIZE = 512
    LEARN_START = 100
    UPDATE_FREQ = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eps = np.finfo(np.float).eps
    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000
    def epsilon_by_frame(frame_idx): return Config.epsilon_final + (
        Config.epsilon_start - Config.epsilon_final)*math.exp(-1. * frame_idx/Config.epsilon_decay)
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'


class CuriousAgent(object):
    def __init__(self, env=None, config=None, log_dir='/tmp/gym'):
        super(CuriousAgent, self).__init__()
        self.config = config
        self.env = env
        self.declare_model()
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)

    def load(self, model_path):
        self.policy.load(torch.load(osp.join(model_path, "model")))
        self.dynamics_model.load(torch.load(osp.join(model_path, "model")))
        self.dynamics_optimizer.load(torch.load(osp.join(model_path, "optim")))
        self.policy_optimizer.load(torch.load(osp.join(model_path, "optim")))

    def save(self, model_path):
        torch.save(self.policy, osp.join(model_path, "model"))
        torch.save(self.dynamics_model, osp.join(model_path, "model"))
        torch.save(self.dynamics_optimizer, osp.join(model_path, "optim"))
        torch.save(self.policy_optimizer, osp.join(model_path, "optim"))

    def declare_model(self):
        self.policy = PolicyConv(
            self.env.observation_space.shape, self.env.action_space.n).to(self.config.device)
        self.dynamics_model = DynamicsModel(
            self.config.FEATURE_SIZE).to(self.config.device)
        self.feature_extractor_model = ConvFeatureExtract(
            self.env.observation_space.shape).to(self.config.device)
        self.dynamics_optimizer = torch.optim.SGD(
            self.dynamics_model.parameters(), lr=self.config.LR)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-2)
        self.criterion = nn.MSELoss()

    def update_dynamics_model(self, loss):
        self.dynamics_optimizer.zero_grad()
        loss.backward()
        self.dynamics_optimizer.step()

    def get_action(self, state):
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.item()

    def update_policy_model(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.policy.rewards[::-1]:
            R = r + self.config.GAMMA * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self.config.eps)
        for log_prob, reward in zip(self.policy.saved_log_probs, rewards):
            policy_loss.append(-log_prob.cuda() * reward.cuda())
        self.policy_optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.policy_optimizer.step()
        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]


class PolicyConv(nn.Module):
    def __init__(self, input_shape, num_actions=4):
        super(PolicyConv, self).__init__()
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.head = nn.Linear(self.feature_size(), num_actions)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        action_scores = self.head(x.view(x.size(0), -1))

        return F.softmax(action_scores, dim=1)

    def feature_size(self):
        return self.conv2(self.conv1(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)


class ConvFeatureExtract(nn.Module):
    def __init__(self, input_shape):
        super(ConvFeatureExtract, self).__init__()
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.head = nn.Linear(self.feature_size(), 512)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.head(x.view(x.size(0), -1))

    def feature_size(self):
        return self.conv2(self.conv1(torch.zeros(1, *self.input_shape))).view(1, -1).size(1)


class DynamicsModel(nn.Module):
    def __init__(self, encoded_state_size):
        super(DynamicsModel, self).__init__()
        self.state_head = nn.Linear(encoded_state_size + 1, encoded_state_size)
        # 1 refers to the action variable, used to pass the loss to the policy network

    def forward(self, state, action):
        action = torch.Tensor([action]).unsqueeze(0).to(config.device)
        next_state_pred = self.state_head(torch.cat([state, action], 1))
        return next_state_pred


if __name__ == "__main__":
    env_id = "PongNoFrameskip-v0"
    log_dir = "./Curiosity/1"
    env = WrapPytorch(gym.make(env_id))
    config = Config()
    env.seed(1)
    torch.manual_seed(1)

    agent = CuriousAgent(env, config, log_dir)

    stored_mses = []
    episode_lengths = []
    print(agent.policy)
    print(agent.dynamics_model)
    print(agent.feature_extractor_model)
    state = env.reset()
    for episode in range(config.MAX_FRAMES):
        frame_count = 0
        while True:
            #env.render()
            last_state = state
            processed_state = torch.tensor(
                [state], dtype=torch.float).to(agent.config.device)
            action = agent.get_action(processed_state)
            state, reward, done, info = env.step(action)

            last_state_encoded = agent.feature_extractor_model(torch.tensor(
                [last_state], dtype=torch.float).to((agent.config.device))).detach()
            state_encoded = agent.feature_extractor_model(
                torch.tensor(state, dtype=torch.float).unsqueeze(0).to((agent.config.device)).detach())

            loss_aggregate = torch.Tensor([0])

            state_encoded_pred = agent.dynamics_model(
                last_state_encoded.to(agent.config.device), action)
            loss = agent.criterion(state_encoded_pred, state_encoded)
            agent.update_dynamics_model(loss)

            loss_aggregate += loss.cpu()
            loss_value = loss.cpu().data.numpy()

            loss_avg = loss_aggregate / (frame_count + 1)
            stored_mses.append(loss_avg)
            agent.policy.rewards.append(loss_avg)

            if done:
                print('Episode {}, loss {}, reward {}'.format(
                    episode, loss_avg.item(), reward))
                agent.update_policy_model()
                state = env.reset()
                agent.writer.add_scalar("data/loss", loss_avg, episode)
                agent.writer.add_scalar("data/reward", reward, episode)
                break
            frame_count += 1
        if episode % 100 == 0:
            agent.save(log_dir)
