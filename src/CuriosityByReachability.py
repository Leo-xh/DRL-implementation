# Simplified implementation of paper https://arxiv.org/pdf/1810.02274.pdf

import sys
import math
import argparse
import gym
import os
import numpy as np
import os.path as osp
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
from torch.distributions import Categorical
from tensorboardX import SummaryWriter

from Wrappers import WrapPyTorch
from collections import namedtuple




class Config(object):
    MEMORY_SIZE = 20000
    ALPHA = 1.0
    BETA = 1.0
    F = max
    GAMMA = 0.99
    k = 800
    BATCH_SIZE = 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EMBED_SIZE = 128
    eps = 1e-6
    MAX_FRAME = 10000


class EmbeddingNet(nn.Module):
    def __init__(self, input_shape, embedded_size=512):
        super(EmbeddingNet, self).__init__()
        self.input_shape = input_shape
        self.embedded_size = embedded_size

        self.conv1 = nn.Conv2d(self.input_shape[0], 32, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        # self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2)
        # self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2)
        self.fc = nn.Linear(self.feature_size(), self.embedded_size)

    def forward(self, state):
        state = F.relu(F.relu(F.relu(self.conv2(F.relu(self.conv1(state))))))
        state = state.view(1,-1)
        return self.fc(state)
    
    def feature_size(self):
        return self.conv2(self.conv1(torch.zeros(1,*self.input_shape))).view(1,-1).size(1)

class MemoryBuffer(object):
    # randomly substitude buffer page, storing embedded state
    def __init__(self, bufferSize):
        self.bufferSize = bufferSize
        self.memory = {} # state:label, the label denotes whether the state is reachable
    
    def sample(self, batch_size):
        # return 2 vectors [s], [id]
        ids = random.choices(list(self.memory.keys()), k=batch_size)
        return [self.memory[id] for id in ids], ids
    
    def store(self, state, id):
        if len(self.memory) > self.bufferSize:
            self.memory.pop(random.choice(list(self.memory.keys())))
        self.memory[id] = state

    def __len__(self):
        return len(self.memory)

class ComparatorNet(nn.Module):
    def __init__(self, embedded_size=512):
        super(ComparatorNet, self).__init__()
        self.embedded_size = embedded_size
        self.fc1 = nn.Linear(self.embedded_size*2, 256)
        # self.fc2 = nn.Linear(1024, 1024)
        # self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, cur_state_embd, buffer_state_embd):
        return F.sigmoid(self.fc4(F.relu(self.fc1(torch.cat([cur_state_embd, buffer_state_embd], 1)))))

class ReachabilityBuffer(object):
    def __init__(self, F):
        self.F = F
    
    def aggregate(self, buffer):
        return self.F(buffer)

class CuriosityModule(object):
    def __init__(self, input_shape, embedded_size=512, memorySize=300000, F=max, alpha=1, beta=1, gamma=1.2, k=1000, log_dir="./tmp/log"):
        """
            refers to the paper
            F: the aggregation function of the reachability
            alpha, beta: hyperparameters of the bonus
            gamma, k: hyperparameters to gap the positive instances and negative instances

            \hat{r_t} = r_t + b(onus)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = 1.2
        self.k = k
        self.log_dir = log_dir

        if not osp.exists(log_dir):
            os.makedirs(log_dir)

        self.embedder = EmbeddingNet(input_shape, embedded_size).to(Config.device)
        self.comparator = ComparatorNet(embedded_size).to(Config.device)
        # print(self.comparator)
        # print(self.embedder)
        self.memoryBuffer = MemoryBuffer(memorySize)
        self.aggregator = ReachabilityBuffer(F)

        self.optimer_embedder = optim.Adam(self.embedder.parameters())
        self.optimer_comp = optim.Adam(self.comparator.parameters())

    def store(self, state, id):
        state_t = self.embedder(state)
        self.memoryBuffer.store(state_t.cpu(), id)    
    
    def get_bonus(self, state):
        states_embd, ids = self.memoryBuffer.sample(Config.BATCH_SIZE)
        cur_state_embd = self.embedder(state)
        dist_buffer = []
        for state_embd in states_embd:
            dist_buffer.append(self.comparator(cur_state_embd, state_embd.to(Config.device)))
        C = self.aggregator.aggregate(dist_buffer)
        return self.alpha*(self.beta-C)
    
    def save(self, *path):
        torch.save(self.embedder.state_dict(), path[0])
        torch.save(self.comparator.state_dict(), path[1])
    
    def load(self, *path):
        self.embedder.load_state_dict(torch.load(path[0]))
        self.comparator.load_state_dict(torch.load(path[1]))


def train_curiosity_module(curi_module, env, log_dir):
    writer = SummaryWriter(osp.join(log_dir, 'log'))
    id = 0
    loss_func = nn.BCELoss()
    optimizer1 = optim.Adam(curi_module.embedder.parameters())
    optimizer2 = optim.Adam(curi_module.comparator.parameters())
 
    obs = env.reset()
    curi_module.memoryBuffer.store(obs,id)
    S = 100
    I = 10000
    for s in range(S):
        for _ in range(200000):
            id += 1
            obs, _, done, _ = env.step(env.action_space.sample())
            curi_module.memoryBuffer.store(obs, id)
            if done:
                obs = env.reset()
                id += 1
                curi_module.memoryBuffer.store(obs, id)
        # prepare data
        for i in range(I):
            states, ids = curi_module.memoryBuffer.sample(Config.BATCH_SIZE//2)
            states_embd = []
            for state in states:
                states_embd.append(curi_module.embedder(torch.tensor(state, device=Config.device, dtype=torch.float).unsqueeze(0)))
            states_embd_1 = []
            states_embd_2 = []
            labels = []
            for _ in range(Config.BATCH_SIZE*4):
                if(len(labels) >= Config.BATCH_SIZE):
                    break
                id1, id2 = random.choices(range(Config.BATCH_SIZE//2),k=2)
                if abs(ids[id1]-ids[id2]) < curi_module.k:
                    labels.append(1)
                elif abs(ids[id1]-ids[id2]) > curi_module.k * curi_module.gamma:
                    labels.append(0)
                else:
                    continue
                states_embd_1.append(states_embd[id1])
                states_embd_2.append(states_embd[id2])

            predit = curi_module.comparator(torch.stack(
                states_embd_1).squeeze(1), torch.stack(states_embd_2).squeeze(1))

            loss = loss_func(predit, torch.tensor(labels, device=Config.device, dtype=torch.float).unsqueeze(1))
            predit_ = [predit.cpu().detach().numpy() > 0.5]
            predit_ = list(map(int, predit))
            acc = sum(np.array(predit_) == np.array(labels))/len(labels)
            print("episode {}, loss {}, accuracy {}".format(s*10+i, loss.cpu().detach().numpy().item(), acc))
            writer.add_scalar("data/loss", loss, s*S+i)
            writer.add_scalar("data/acc", acc, s*S+i)
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
        curi_module.save(osp.join(log_dir, "embedder_{}_{}".format(s, i)), osp.join(log_dir, "comp_{}_{}".format(s, i)))


class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(ActorCritic, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 8, kernel_size=3, stride=2)
        # self.conv4 = nn.Conv2d(64, 32, kernel_size=3, stride=2)
        
        self.value_head = nn.Linear(self.feature_size(), 1)
        self.action_head = nn.Linear(self.feature_size(), self.num_actions)
    
    def forward(self, state):
        state = F.relu(self.conv3(F.relu(self.conv2(F.relu(self.conv1((state)))))))
        values = self.value_head(state.view(1,-1))
        action_scores = self.action_head(state.view(1,-1))
        return F.softmax(action_scores, dim=-1), values

    def feature_size(self):
        return self.conv3(self.conv2(self.conv1(torch.zeros(1, *self.input_shape)))).view(1, -1).size(1)
    
# TODO: simultaneously train the curiosity module and the actor-critic module
class Agent(object):
    SavedAction = namedtuple("SavedAction", field_names=["value", "log_prob"])
    def __init__(self, env=None, log_dir='./tmp/curiosity-reachability/1'):
        self.ac_net = ActorCritic(env.observation_space.shape, env.action_space.n).to(Config.device)
        self.curi_module = CuriosityModule(env.observation_space.shape, Config.EMBED_SIZE, Config.MEMORY_SIZE, Config.F, Config.ALPHA, Config.BETA, Config.GAMMA, Config.k, log_dir)
        self.optimer_ac = optim.Adam(self.ac_net.parameters(), lr=2e-7)
        
        self.saved_actions = []
        # \hat{r} = r + b
        self.returns = []
        # self.next_states = []

        # special data structure

        self.writer = SummaryWriter(osp.join(log_dir,"log"))

    def get_action(self, state):
        # print(state)
        probs, state_value = self.ac_net(state)
        # print(probs, state_value)
        action_selector = Categorical(probs=probs.view(1,-1))
        action = action_selector.sample()
        self.saved_actions.append(Agent.SavedAction(state_value, action_selector.log_prob(action)))
        return action.item()
    
    def update_episode(self):
        loss = self.compute_loss()
        # if the curiosity module needs no training
        self.optimer_ac.zero_grad()
        loss.backward()
        self.optimer_ac.step()
        return loss.item()

    # def store_next_state(self, n_state):
        # self.next_states.append(n_state)
    
    def get_bonus(self, state):
        return self.curi_module.get_bonus(state)

    def compute_loss(self):
        policy_losses = []
        value_losses = []
        rewards = []
        R = 0
        for r in self.returns:
            R = r + Config.GAMMA*R
            # print(r, R)
            rewards.insert(0,R)
        rewards = torch.tensor(rewards, device=Config.device, dtype=torch.float)
        # compute advantages
        rewards = rewards - rewards.mean()
        # print("len of self.saved_actions {} len of rewards {}".format(len(self.saved_actions), len(rewards)))
        for (value, log_prob), R in zip(self.saved_actions, rewards):
            # print(value, R, log_prob)
            advantage = R - value.item()
            policy_losses.append(-log_prob*advantage)
            value_losses.append((value-torch.tensor([R],device=Config.device).pow(2)))
        # print("shape of policy_losses {}".format(torch.tensor(
            # policy_losses, device=Config.device, dtype=torch.float).shape))
        del self.saved_actions[:]
        del self.returns[:]
        # print(policy_losses)
        # print(value_losses)
        return torch.stack(policy_losses).mean() + torch.stack(value_losses).mean()

    def save(self, *path):
        torch.save(self.ac_net, path[0])
        self.curi_module.save(path[1:])

    def load(self, *path):
        self.ac_net.load_state_dict(torch.load(path[0]))
        self.curi_module.load(path[1:])

def main_train_curi():
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    # torch.cuda.set_device(2)
    env_id = "PongNoFrameskip-v0"
    log_dir = "./Curiosity-reachability/1"
    env = gym.make(env_id)
    env = WrapPyTorch(env)
    curi_module = CuriosityModule(env.observation_space.shape, log_dir=log_dir)
    train_curiosity_module(curi_module, env, log_dir)

def main_train_agent():
    env_id = "Pong-v4"
    env = WrapPyTorch(gym.make(env_id))
    agent = Agent(env)
    frames = 0
    for ep in range(Config.MAX_FRAME):
        ep_reward, obs = 0, env.reset()
        while(True):
            obs_t = torch.tensor(obs, device=Config.device, dtype=torch.float).unsqueeze(0)
            agent.curi_module.store(obs_t, frames)
            action = agent.get_action(obs_t)
            obs, reward, done, _ = env.step(action)
            if frames >= Config.MEMORY_SIZE:
                bonus = agent.get_bonus(obs_t)
                agent.returns.append(reward+bonus)
            else:
                agent.returns.append(reward)
            ep_reward += reward
            frames += 1
            if done:
                break
        loss = agent.update_episode()
        print("episode {} reward {} loss {}".format(ep, ep_reward, loss))
        agent.writer.add_scalar("data/loss", loss, ep)
        agent.writer.add_scalar("data/reward", ep_reward, ep)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '2'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    torch.cuda.set_device(2)
    main_train_agent()
