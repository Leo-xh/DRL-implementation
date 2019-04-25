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

from ReplayMemory import ExperienceReplayMemory
from WrapPytorch import WrapPytorch

class Config(object):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epsilon_start = 1.0
    epsilon_final = 0.01
    epsilon_decay = 30000
    epsilon_by_frame = lambda frame_idx: Config.epsilon_final + (
        Config.epsilon_start - Config.epsilon_final)*math.exp(-1. * frame_idx/Config.epsilon_decay)

    GAMMA = 0.99
    LR = 1e-4

    TARGET_NET_UPDATE_FREQ = 1000
    EXP_REPLAY_SIZE = 10000
    BATCH_SIZE = 32

    LEARN_START = 100
    MAX_FRAMES = 1000000
    UPDATE_FREQ = 1


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, self.num_actions)

    def forward(self, x):
        if torch.cuda.is_available():
            x = x.cuda()
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1,-1).size()[1]

class DQNAgent(object):
    def __init__(self, env, log_dir='./dqn/1'):
        self.device = Config.device
        self.gamma = Config.GAMMA
        self.lr = Config.LR
        self.target_net_update_freq = Config.TARGET_NET_UPDATE_FREQ
        self.experience_replay_size = Config.EXP_REPLAY_SIZE
        self.batch_size = Config.BATCH_SIZE
        self.learn_start = Config.LEARN_START
        self.update_freq = Config.UPDATE_FREQ

        self.env = env
        self.num_feats = env.observation_space.shape
        self.num_actions = env.action_space.n

        
        self.log_dir = log_dir
        self.writer = SummaryWriter(self.log_dir)
        
        self.declare_networks()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)
        self.target_model.to(self.device)

        self.update_count = 0
        self.declare_memory()

    def declare_networks(self):
        self.model = DQN(self.num_feats, self.num_actions)
        self.target_model = DQN(self.num_feats, self.num_actions)
        
    def declare_memory(self):
        self.memory = ExperienceReplayMemory(self.experience_replay_size)
    
    def store_to_replay(self, s, a, r, s_):
        self.memory.push((s,a,r,s_))
    
    def sample_memory(self):
        transitions, indices, weights = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,) + self.num_feats

        batch_state = torch.tensor(batch_state, device=self.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(batch_action, device=self.device, dtype=torch.long).squeeze().view(-1,1)
        batch_reward = torch.tensor(batch_reward, device=self.device, dtype=torch.float).squeeze().view(-1,1)
        batch_next_state = torch.tensor(batch_next_state, device=self.device, dtype=torch.float).view(shape)

        return batch_state, batch_action, batch_reward, batch_next_state, indices, weights


    def get_action(self, s, eps=0.2):
        with torch.no_grad():
            if np.random.random() >= eps:
                X = torch.tensor([s], device=self.device, dtype=torch.float)
                if torch.cuda.is_available():
                    a = self.model(X).cpu().max(1)[1].view(1,1)
                else:
                    a = self.model(X).max(1)[1].view(1,1)
                return a.item()
            else:
                return np.random.randint(0, self.num_actions)
    
    def get_loss(self, batch_vars):
        batch_state, batch_action, batch_reward, batch_next_state, indices, weights = batch_vars

        q_values = self.model(batch_state).gather(1, batch_action)
        target_q_values = batch_reward + self.gamma*self.target_model(batch_next_state).detach().max(1)[0]

        loss = self.loss_func(q_values, target_q_values)
        return loss

    def update(self, s, a, r, s_, frame_idx=0):
        self.store_to_replay(s,a,r,s_)
        if frame_idx < self.learn_start or frame_idx % self.update_freq != 0:
            return None
        batch_vars = self.sample_memory()
        loss = self.get_loss(batch_vars)
        self.writer.add_scalar("data/loss", loss, ep)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_target()
    
    def update_target(self):
        self.update_count += 1
        self.update_count = self.update_count % self.target_net_update_freq
        if self.update_count == 0:
            self.target_model.load_state_dict(self.model.state_dict())
    
    def loss_func(self, predict, target):
        return nn.MSELoss()(predict, target)

    def save(self, model_path):
        torch.save(self.model.state_dict(), osp.join(model_path,"model_weight"))
        torch.save(self.optimizer.state_dict(), osp.join(model_path, "optim_weight"))

    def load(self, model_path):
        self.model.load_state_dict(torch.load(
            osp.join(model_path, "model_weight")))
        self.target_model.load_state_dict(
            torch.load(osp.join(model_path, "model_weight")))
        self.optimizer.load_state_dict(torch.load(
            osp.join(model_path, "optim_weight")))


if __name__ == "__main__":
    log_dir = "./dqn/4"
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    env_id = "Boxing-v0"
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
