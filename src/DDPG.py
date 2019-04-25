
'''
Implementation of Deep Deterministic Policy Gradients (DDPG) with pytorch on Gym game
Original paper: https://arxiv.org/abs/1509.02971
'''

import argparse
import os
import os.path as osp
import numpy as np
import random

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from itertools import count
from torch.distributions import Normal
from tensorboardX import SummaryWriter
from Wrappers import WrapPyTorch
from gym.wrappers import Monitor
from ReplayMemory import ExperienceReplayMemory
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="train", type=str, help="mode")
    parser.add_argument("--num", default=0, type=int, help="the index of the loading model")
    parser.add_argument("--env_name", default="CarRacing-v0", type=str, help="gym environment name")
    parser.add_argument("--tau", default=0.005, type=float, help="target smoothing coefficient")
    parser.add_argument("--test_iteration", default=10, type=int, help="num of game for testing")
    parser.add_argument("--train_iteration", default=1000000, type=int, help="num of game for training")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--gamma", default=0.99, type=float, help="discounted factor")
    parser.add_argument("--capacity", default=10000, type=int, help="replay buffer size")
    parser.add_argument("--batch_size", default=64, type=int, help="training batch size")
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument('--sample_frequency', default=256, type=int)
    parser.add_argument('--render', action="store_true", help="show UI or not")
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--load', action="store_true")
    parser.add_argument('--render_interval', default=100, type=int, help="after render_interval, the env.render() will work")
    parser.add_argument('--exploration_noise', default=0.1, type=float)
    parser.add_argument('--max_episode', default=2000, type=int)
    parser.add_argument('--print_log', default=5, type=int)
    parser.add_argument('--update_iteration', default=1, type=int)
    parser.add_argument("--log_dir", default="./tmp/DDPG/5", type=str)

    return parser.parse_args()

args = get_args()
args.device = ("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, input_shape, action_shape, action_lim):
        super(Actor, self).__init__()
        self.input_shape = input_shape
        self.action_shape = action_shape
        self.action_lim = action_lim

        self.nn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(True), nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(True), nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(True), nn.Conv2d(32, 16, kernel_size=3, stride=2),)
        self.nn_fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, action_shape),
            nn.Sigmoid()
        )

    def forward(self, state):
        x = self.nn(state)
        x = self.nn_fc(x.view(x.size(0),-1))
        # TODO: lim
        # action_high_t = torch.tensor(self.action_lim[0], device=args.device, dtype=torch.float)
        # action_low_t = torch.tensor(self.action_lim[1], device=args.device, dtype=torch.float)
        
        # x[x > 0] *= action_high_t.repeat(x.size(0), 1)[x > 0]
        # x[x < 0] *= -action_high_t.repeat(x.size(0), 1)[x < 0]
        return x

    def feature_size(self):
        return self.nn(torch.zeros(1, *self.input_shape)).view(1,-1).size(1)

class Critic(nn.Module):
    def __init__(self, input_shape, action_shape):
        super(Critic, self).__init__()
        self.action_shape = action_shape
        self.input_shape = input_shape
        self.nn = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=2),
            nn.ReLU(True), nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(True), nn.Conv2d(32, 16, kernel_size=3, stride=2))
        self.nn_fc = nn.Sequential(
            nn.Linear(self.feature_size()+self.action_shape, 512),
            nn.ReLU(True),
            nn.Linear(512, 1)
        )

    def forward(self, state, action):
        x = self.nn(state)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], 1)
        x = self.nn_fc(x)

        return x

    def feature_size(self):
        return self.nn(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

class DDPGAgent(object):
    def __init__(self, env):
        self.input_shape = env.observation_space.shape
        self.action_shape = env.action_space.shape[0]
        self.action_lim = [env.action_space.high, env.action_space.high]
        self.actor = Actor(self.input_shape, self.action_shape, self.action_lim).to(args.device)
        self.actor_target = Actor(
            self.input_shape, self.action_shape, self.action_lim).to(args.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), args.lr)

        self.critic = Critic(self.input_shape, self.action_shape).to(args.device)
        self.critic_target = Critic(self.input_shape, self.action_shape).to(args.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), args.lr)

        self.replay_buffer = ExperienceReplayMemory(args.capacity)
        self.writer = SummaryWriter(args.log_dir)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0

    def select_action(self, state):
        state = torch.tensor(state, device=args.device, dtype=torch.float).unsqueeze(0)
        return self.actor(state).cpu().detach().numpy().flatten()
    
    def sample_memory(self):
        transitions, indices, weights = self.replay_buffer.sample(args.batch_size)

        batch_state, batch_action, batch_reward, batch_next_state = zip(*transitions)

        shape = (-1,) + self.input_shape

        batch_state = torch.tensor(
            batch_state, device=args.device, dtype=torch.float).view(shape)
        batch_action = torch.tensor(
            batch_action, device=args.device, dtype=torch.float).squeeze().view(-1, self.action_shape)
        batch_reward = torch.tensor(
            batch_reward, device=args.device, dtype=torch.float).squeeze().view(-1, 1)
        batch_next_state = torch.tensor(
            batch_next_state, device=args.device, dtype=torch.float).view(shape)

        return batch_state, batch_action, batch_reward, batch_next_state, indices, weights

    def update(self):
        for it in range(args.update_iteration):
            b_s, b_a, b_r, b_ns, _, _ = self.sample_memory()

            target_Q = self.critic_target(b_ns, self.actor_target(b_ns))
            target_Q = b_r + args.gamma * target_Q.detach()

            current_Q = self.critic(b_s, b_a)

            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar("data/loss_critic", critic_loss, self.num_critic_update_iteration)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(b_s, self.actor(b_s)).mean()
            self.writer.add_scalar(
                "data/actor_loss", actor_loss, self.num_actor_update_iteration)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), osp.join(args.log_dir, str(self.num_actor_update_iteration) + '_actor.pth'))
        torch.save(self.critic.state_dict(), osp.join(args.log_dir, str(self.num_critic_update_iteration) +'_critic.pth'))
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self, num1, num2):
        self.actor.load_state_dict(torch.load(osp.join(args.log_dir, str(num1) + '_actor.pth')))
        self.critic.load_state_dict(torch.load(osp.join(args.log_dir, str(num2) + '_critic.pth')))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


if __name__ == "__main__":
    if not osp.exists(args.log_dir):
        os.makedirs(args.log_dir)
    env = WrapPyTorch(gym.make(args.env_name))
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = DDPGAgent(env)
    ep_r = 0
    if args.mode == 'test':
        agent.load(args.num, args.num)
        for i in range(args.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t >= args.max_episode:
                    print(
                        "Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state
    else :
        if args.load:
            agent.load(args.num, args.num)
        for i in range(args.train_iteration):
            state = env.reset()
            ep_r = 0
            for t in range(args.max_episode):
                action = agent.select_action(state)
                action += action+np.random.uniform(-0.2, 0.2, *action.shape)
                action[action > 1] = 1
                action[action < 0] = 0

                # print(action)
                next_state, reward, done, info = env.step(action)
                # print(reward)
                ep_r += reward
                if args.render:
                    env.render()
                agent.replay_buffer.push((state, action, reward, next_state))
                state = next_state
                if len(agent.replay_buffer) >= args.capacity:
                    agent.update()
                if done or t > args.max_episode:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    break

            print("Ep_i \t{}, the ep_r is \t{:0.2f}".format(i, ep_r))
            if i % args.log_interval == 0:
                agent.save()
