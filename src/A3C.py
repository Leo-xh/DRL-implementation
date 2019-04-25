import sys
import math
import argparse
import gym
import os
import argparse
import numpy as np
import os.path as osp
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.multiprocessing as mp
import torch.optim as optim

from torch.distributions import Categorical
from tensorboardX import SummaryWriter
from functools import reduce
from Wrappers import WrapPyTorch
from gym.wrappers import Monitor

def get_args():
    parser = argparse.ArgumentParser(description="argements for A3C")
    parser.add_argument('--env', default='PongNoFrameskip-v4',
                        type=str, help='gym environment')
    parser.add_argument('--processes', default=10, type=int,
                        help='number of processes to train with')
    parser.add_argument('--render', default=False, type=bool,
                        help='renders the atari environment')
    parser.add_argument('--test', action="store_true",
                        help='sets lr=0, chooses most likely actions')
    parser.add_argument('--rnn_steps', default=20, type=int,
                        help='steps to train LSTM over')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed random # generators (for reproducibility)')
    parser.add_argument('--gamma', default=0.99, type=float,
                        help='rewards discount factor')
    parser.add_argument('--tau', default=1.0, type=float,
                        help='generalized advantage estimation discount')
    parser.add_argument('--horizon', default=0.99, type=float,
                        help='horizon for running averages')
    parser.add_argument('--hidden', default=256, type=int,
                        help='hidden size of GRU')
    parser.add_argument('--frames', default=10000000, type=int,
                        help='max frames')
    return parser.parse_args()

args = get_args()


class ActorCritic(nn.Module):
    def __init__(self, input_shape, num_actions, memory_size):
        super(ActorCritic, self).__init__()
        self.nn = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 32, kernel_size=3, stride=2),
            nn.ReLU(True),
        )
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.rnn = nn.GRUCell(self.feature_size(), memory_size)
        self.value = nn.Linear(memory_size, 1)
        self.action = nn.Linear(memory_size, num_actions)

    def forward(self, input_, hidden):
        x = self.nn(input_)
        hx = self.rnn(x.view(x.size(0), -1), hidden)
        return self.value(hx), self.action(hx), hx
    
    def feature_size(self):
        return self.nn(torch.zeros(1, *self.input_shape)).view(1,-1).size(1)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)
    
    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))


class SharedAdam(optim.Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['shared_steps'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                self.state[p]['shared_steps'] += 1
                # a "step += 1"  comes later
                self.state[p]['step'] = self.state[p]['shared_steps'] - 1
        super(SharedAdam, self).step(closure)
            
class A3C_Agent(object):
    def __init__(self, env=None, log_dir='/tmp/gym'):
        self.env = env
        self.log_dir = log_dir

        self.shared_model = ActorCritic(env.observation_space.shape, env.action_space.n, args.hidden).to(args.device).share_memory()
        self.shared_optimizer = SharedAdam(self.shared_model.parameters(), lr=args.lr)

    def save(self, *path):
        torch.save(self.shared_model.state_dict(), path[0])
        torch.save(self.shared_optimizer.state_dict(), path[1])
    
    def load(self, *path):
        self.shared_model.load_state_dict(torch.load(path[0]))
        self.shared_optimizer.load_state_dict(torch.load(path[1]))

def get_loss(args, values, logps, actions, rewards):
    values_np = values.detach().cpu().numpy()

    log_action = logps.gather(1, actions.unsqueeze(1)).view(1, -1)
    delta_t = np.asarray(rewards) + args.gamma * \
        values_np[1:] - values_np[:-1]
    gae = reduce(lambda x, y: x*args.gamma * args.tau + y, reversed(delta_t), 0)
    policy_loss = -(log_action.view(-1)*torch.tensor(gae, device=args.device, dtype=torch.float32)).mean()

    discounted_r = reduce(lambda x, y: x*args.gamma+y,
                            reversed(rewards), 0)
    discounted_r_t = torch.tensor(
        discounted_r, device=args.device, dtype=torch.float32)
    # values contains the estimation of the finished state
    value_loss = (discounted_r-values.view(-1)[:-1]).pow(2).mean()

    entropy_loss = (-log_action*torch.exp(log_action)).sum()

    return policy_loss + 0.5*value_loss - 0.01*entropy_loss

def train(shared_model, shared_optimizer, rank, info, args):
    env = gym.make(args.env)
    env = WrapPyTorch(env)
    if args.test == True:
        env = Monitor(env, directory=osp.join(args.log_dir, "log"))

    env.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    input_shape, action_nums = env.observation_space.shape, env.action_space.n
    model = ActorCritic(input_shape, action_nums, args.hidden).to(args.device)
    state = torch.tensor(env.reset(), device=args.device, dtype=torch.float).unsqueeze(0)

    episode_length, ep_reward, ep_loss, done = 0, 0, 0, True

    while info['frames'][0] <= args.frames or args.test:
        model.load_state_dict(shared_model.state_dict())

        hx = torch.zeros(1, args.hidden, device=args.device)
        values, log_actions, actions, rewards = [], [], [], []

        for step in range(args.rnn_steps):
            episode_length += 1
            value, logit, hx = model(state, hx)
            logp = F.log_softmax(logit, dim=-1)

            action_selector = Categorical(probs=logp)
            action = action_selector.sample()
            state, reward, done, _ = env.step(action.cpu().numpy()[0])
            if args.render:
                env.render()

            state = torch.tensor(state, device=args.device, dtype=torch.float).unsqueeze(0)
            ep_reward += reward
            reward = np.clip(reward, -1, 1)  # reward
            done = done or episode_length >= 1e4  # don't playing one ep for too long

            info['frames'].add_(1)
            num_frames = int(info['frames'].item())
            if num_frames % 2e6 == 0:  # save every 2M frames
                print('\n\t{:.0f}M frames: saved model\n'.format(
                    num_frames/1e6))
                model.save(osp.join(args.log_dir, 'model.{:.0f}.tar'.format(num_frames/1e6)))

            if done:  # update shared data
                info['episodes'] += 1
                interp = 1 if info['episodes'][0] == 1 else 1 - \
                    args.horizon
                info['run_reward'].mul_(1-interp).add_(interp * ep_reward)
                info['run_loss'].mul_(1-interp).add_(interp * ep_loss)

            if rank == 0:  # print info ~ every minute
                print('episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                        .format(info['episodes'].item(), num_frames/1e6,
                                info['run_reward'].item(), info['run_loss'].item()))

            if done:  # maybe print info.
                episode_length, ep_reward, ep_loss = 0, 0, 0
                state = torch.tensor((env.reset()), device=args.device, dtype=torch.float).unsqueeze(0)

            values.append(value)
            log_actions.append(logp)
            actions.append(action)
            rewards.append(reward)
        next_value = torch.zeros(1, 1).to(args.device) if done else model(
            state, hx)[0]
        values.append(next_value.detach())
        loss = get_loss(args, torch.cat(values).to(args.device), torch.cat(
            log_actions).to(args.device), torch.cat(actions).to(args.device), np.asarray(rewards))
        ep_loss += loss.item()
        shared_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 40)

        for param, shared_param in zip(model.parameters(), shared_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad  # sync gradients with shared model
        shared_optimizer.step()



if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    mp.set_start_method('spawn')
    args.log_dir = './A3C/2'
    torch.manual_seed(args.seed)
    args.device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")
    if not osp.exists(args.log_dir):
        os.makedirs(args.log_dir)
    env = WrapPyTorch(gym.make(args.env))
    if args.test:
        args.processes = 1
        args.render = True
    agent = A3C_Agent(env, args.log_dir)
    processes = []
    info = {k: torch.tensor([0], device=args.device, dtype=torch.float32).share_memory_()for k in [
        'run_reward', 'run_loss', 'episodes', 'frames']}
    print("Starting workers...")
    for rank in range(args.processes):
        p = mp.Process(target=train, args=(agent.shared_model, agent.shared_optimizer, rank, info, args))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

