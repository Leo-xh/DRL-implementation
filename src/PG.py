import gym
import math
import numpy as np
import os
import os.path as osp
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from ReplayMemory import ExperienceReplayMemory
from WrapPytorch import WrapPytorch

class Config(object):
    GAMMA = 0.99
    LR = 0.002
    ENTROPY_BETA = 0.01
    MAX_FRAMES = 1000000

    device = ("cuda" if torch.cuda.is_available() else "cpu")

class PG_NN(nn.Module):
    def __init__(self, input_shape, action_dim, action_lim):
        '''
            action_lim: (2,action_dim)
        '''
        super(PG_NN, self).__init__()
        self.input_shape = input_shape
        self.action_dim = action_dim
        self.action_lim = action_lim
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.nn = nn.Sequential(
            nn.Linear(self.feature_size(), 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        if torch.cuda.is_available():
            action_range = torch.tensor(self.action_lim[0]-self.action_lim[1], device=Config.device, dtype=torch.float).view(self.action_dim)
            x = x.cuda()
            x = self.conv2(self.conv1(x))
            x = x.view(x.size(0), -1)
            return self.nn(x.cuda())*action_range
        x = x.float()
        x = self.conv2(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.nn(x.float())

        return x

    def feature_size(self):
        return self.conv2(self.conv1(torch.zeros(1,*self.input_shape))).view(1,-1).size(1)

class PGAgent(object):
    def __init__(self, env, log_dir='./pg/1'):
        self.env = env
        self.log_dir = log_dir
        self.gamma = Config.GAMMA
        self.lr = Config.LR
        self.beta = Config.ENTROPY_BETA
        self.baseline = deque(maxlen=100000)
        self.device = Config.device

        self.declare_memory()
        self.declare_network()

        self.writer = SummaryWriter(log_dir=self.log_dir)

    def discounted_rewards(self, memories):
        disc_ret = np.zeros(len(memories))
        run_add = 0
        for t in reversed(range(len(memories))):
            if memories[t][-1]==True:
                run_add = 0
            run_add = run_add * self.gamma + memories[t][2]
            disc_ret[t] = run_add
        return disc_ret
    
    def get_action(self, s):
        if torch.cuda.is_available():
            return self.model(s).detach().cpu().view(self.env.action_space.shape)
        return self.model(s).cpu().view(self.env.action_space.shape)

    def declare_network(self):
        self.model = PG_NN(self.env.observation_space.shape, self.env.action_space.shape[0], [self.env.action_space.high,self.env.action_space.low])
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

    def declare_memory(self):
        self.memory = []

    def clear_memory(self):
        self.memory = []

    def store(self, s, a, r, s_, done):
        self.memory.append((s,a,r,s_,done))
    
    def update(self):
        loss = self.get_loss(self.memory)
        self.writer.add_scalar("data/loss", loss, it)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.clear_memory()

    def get_loss(self, memory_vars):
        disc_rew = self.discounted_rewards(memory_vars)
        self.baseline.extend(disc_rew)
        disc_rew -= np.mean(self.baseline)
        acts = self.model(torch.tensor([e[0] for e in memory_vars]))
        disc_rew_t = torch.tensor(disc_rew, dtype=torch.float, device=self.device)
        log_softmax_t = torch.log(acts).view(-1,1)
        loss = -torch.mean(log_softmax_t*disc_rew_t).squeeze()

        return loss

    def load(self, model_path):
        self.model.load(torch.load(osp.join(model_path, "model")))
        self.optimizer.load(torch.load(osp.join(model_path, "optim")))

    def save(self, model_path):
        torch.save(self.model, osp.join(model_path,"model"))
        torch.save(self.optimizer, osp.join(model_path,"optim"))


if __name__ == "__main__":
    env_id = "CarRacing-v0"
    log_dir = "./pg/1"
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    env = gym.make(env_id)
    # env = gym.wrappers.Monitor(env, osp.join(log_dir))
    env = WrapPytorch(env)

    agent = PGAgent(env, log_dir)
    obs = env.reset()
    it = 0
    for ep in range(Config.MAX_FRAMES):
        # env.render()
        action = agent.get_action(torch.tensor([obs], dtype=torch.float, device=agent.device)).detach().numpy()
        print(action)
        prev_obs = obs
        obs, reward, done, _ = env.step(action)
        agent.store(prev_obs, action, reward, obs, done)
        if done:
            agent.writer.add_scalar("data/reward", reward, it)
            it += 1
            agent.update()
            if it % 10 == 0:
                agent.save(log_dir)
            print("episode", it, "reward:", reward)
            obs = env.reset()

    env.close()
    agent.writer.close()
