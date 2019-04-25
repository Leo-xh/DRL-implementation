import argparse
import datetime
import math
import os
import os.path as osp
from collections import deque, namedtuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers import Monitor
from tensorboardX import SummaryWriter
from torch.nn.utils.clip_grad import clip_grad_norm_


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="BipedalWalker-v2", type=str)
    parser.add_argument("--max_iteration", default=1000000, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--ppo_epoch", default=7, type=int)
    parser.add_argument("--device", default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument("--clip_gradient", default=0.2, type=float)
    parser.add_argument("--clip_eps", default=0.2, type=float)
    parser.add_argument("--trajectory_size", default=2000, type=int)
    parser.add_argument("--gae_lambda", default=0.95, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--test_episode", default=10, type=int)
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--log_interval", default=10, type=int)
    parser.add_argument("--policy_lr", default=0.0004, type=float)
    parser.add_argument("--value_lr", default=0.001, type=float)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--num", default=0, type=int)
    parser.add_argument("--log_dir", default="../tmp/PPO/1", type=str)

    return parser.parse_args()

args = get_args()
Memory = namedtuple('Memory', ['obs', 'action', 'new_obs',
                               'reward', 'done', 'value', 'adv'], verbose=False, rename=False)



class A2C_policy(nn.Module):
    '''
    Policy neural network
    '''

    def __init__(self, input_shape, n_actions):
        super(A2C_policy, self).__init__()

        self.lp = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU())

        self.mean_l = nn.Linear(32, n_actions[0])
        self.mean_l.weight.data.mul_(0.1)

        self.var_l = nn.Linear(32, n_actions[0])
        self.var_l.weight.data.mul_(0.1)

        self.logstd = nn.Parameter(torch.zeros(n_actions[0]).type(torch.float))

    def forward(self, x):
        ot_n = self.lp(x.float())
        return F.tanh(self.mean_l(ot_n))


class A2C_value(nn.Module):
    '''
    Actor neural network
    '''

    def __init__(self, input_shape):
        super(A2C_value, self).__init__()

        self.lp = nn.Sequential(
            nn.Linear(input_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.lp(x.float())

class PPO_Agent(object):
    game_rew = 0
    last_game_rew = 0
    game_n = 0
    last_games_rews = [-200]
    n_iter = 0
    
    def __init__(self, env, n_steps, gamma, gae_lambda):
        super(PPO_Agent, self).__init__()

        self.env = env
        self.obs = self.env.reset()

        self.n_steps = n_steps
        self.action_n = self.env.action_space.shape
        self.observation_n = self.env.observation_space.shape[0]
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.policy_net = A2C_policy(self.observation_n, self.action_n).to(args.device)
        self.value_net = A2C_value(self.observation_n).to(args.device)

    def get_action(self, obs):
        action = self.policy_net(torch.tensor(obs, device=args.device, dtype=torch.float))
        action = action.detach().cpu().numpy().squeeze()
        action = np.clip(action, -1, 1)
        return action

    def steps(self, device=args.device):
        memories = []
        for s in range(self.n_steps):
            self.n_iter += 1
            ag_mean = self.policy_net(torch.tensor(self.obs, device=device))

            logstd = self.policy_net.logstd.detach().cpu().numpy()
            action = ag_mean.detach().cpu().numpy() + np.exp(logstd) * np.random.normal(size=logstd.shape)
            action = np.clip(action, -1, 1)

            state_value = float(self.value_net(torch.tensor(self.obs, device=device, dtype=torch.float)))

            n_obs, reward, done, _ = self.env.step(action)

            if done:
                memories.append(Memory(obs=self.obs, action=action, new_obs=n_obs, reward=0, done=done, value=state_value, adv=0))
            else :
                memories.append(Memory(obs=self.obs, action=action, new_obs=n_obs,reward=reward, done=done, value=state_value, adv=0))
            
            self.game_rew += reward
            self.obs = n_obs

            if done:
                print("episode {}: {} steps reward {} mean reward {}".format(self.game_n, self.n_iter, self.game_rew, np.mean(self.last_games_rews[-100:])))

                self.obs = self.env.reset()
                self.last_game_rew = self.game_rew
                self.game_rew = 0
                self.game_n += 1
                self.n_iter = 0
                self.last_games_rews.append(self.last_game_rew)
        
        return self.get_gae(memories)

    
    def get_gae(self, memories):
        upd_memories = []
        run_add = 0

        for t in reversed(range(len(memories)-1)):
            if memories[t].done:
                run_add = memories[t].reward
            else:
                sigma = memories[t].reward + self.gamma * memories[t+1].reward - memories[t].value
                run_add = sigma + run_add * self.gamma * self.gae_lambda

            upd_memories.append(Memory(obs=memories[t].obs, action=memories[t].action, new_obs=memories[t].new_obs,reward=run_add + memories[t].value, done=memories[t].done, value=memories[t].value, adv=run_add))

        return upd_memories[::-1]
    
    def log_policy_prob(self, mean, std, actions):
        # policy log probability
        act_log_softmax = -((mean-actions)**2)/(2*torch.exp(std).clamp(min=1e-4)) - torch.log(torch.sqrt(2*math.pi*torch.exp(std)))
        return act_log_softmax


    def compute_log_policy_prob(self, memories, device):
        n_mean = self.policy_net(torch.tensor(np.array([m.obs for m in memories], dtype=np.float)).to(device))
        logstd = self.policy_net.logstd
        actions = torch.tensor(np.array([m.action for m in memories]), dtype=torch.float, device=device)
        return self.log_policy_prob(n_mean, logstd, actions)


    def clipped_PPO_loss(self, memories, old_log_policy, adv, epsilon, writer, device):
        rewards = torch.tensor(np.array([m.reward for m in memories], dtype=np.float32)).to(device)
        value = self.value_net(torch.tensor(np.array([m.obs for m in memories], dtype=np.float32)).to(device))

        value_loss = F.mse_loss(value.squeeze(-1), rewards)

        new_log_policy = self.compute_log_policy_prob(memories, device)
        rt_theta = torch.exp(new_log_policy - old_log_policy.detach())

        adv = adv.unsqueeze(-1)
        policy_loss = -torch.mean(torch.min(rt_theta*adv, torch.clamp(rt_theta, 1-epsilon, 1+epsilon)*adv))

        return policy_loss, value_loss


    def test_game(self, test_episodes, test_env):
        reward_games = []
        step_games = []
        for _ in range(test_episodes):
            obs = test_env.reset()
            ep_reward = 0
            ep_step = 0
            while True:
                action = self.policy_net(obs)
                n_obs, r, done, _ = test_env.step(action)
                ep_step += 1
                obs = n_obs
                ep_reward += r

                if done:
                    reward_games.append(ep_reward)
                    step_games.append(ep_step)
                    obs = test_env.reset()
                    break

        return np.mean(reward_games), np.mean(step_games)

if __name__ == "__main__":
    env = gym.make(args.env_name)
    writer = SummaryWriter(args.log_dir)

    best_test_result = -1e5

    if args.save_video:
        env = Monitor(env, osp.join(args.log_dir, "Videos"), video_callable=lambda episode_id:episode_id % 10 == 0)
    
    agent = PPO_Agent(env, args.trajectory_size, args.gamma, args.gae_lambda)

    optimizer_policy = optim.Adam(agent.policy_net.parameters(), lr=args.policy_lr)
    optimizer_value = optim.Adam(agent.value_net.parameters(), lr=args.value_lr)

    if args.load:
        print("loading models")
        checkpoint = torch.load(osp.join(args.log_dir, args.num+".pth.tar"))
        agent.policy_net.load_state_dict(checkpoint['agent_policy'])
        agent.value_net.load_state_dict(checkpoint['agent_value'])
        optimizer_policy.load_state_dict(checkpoint['optimizer_policy'])
        optimizer_value.load_state_dict(checkpoint['optimizer_value'])

    experience = []
    n_iter = 0

    while n_iter < args.max_iteration:
        n_iter += 1

        batch = agent.steps()
        old_log_policy = agent.compute_log_policy_prob(batch, args.device)

        batch_adv = np.array([m.adv for m in batch])
        
        batch_adv = (batch_adv - np.mean(batch_adv)) / (np.std(batch_adv) + 1e-7)
        batch_adv = torch.tensor(batch_adv, device=args.device, dtype=torch.float)

        pol_loss_acc = []
        val_loss_acc = []

        for s in range(args.ppo_epoch):
            for mb in range(0, len(batch), args.batch_size):
                mini_batch = batch[mb:mb+args.batch_size]
                minib_old_log_policy = old_log_policy[mb:mb+args.batch_size]
                minib_adv = batch_adv[mb:mb+args.batch_size]

                pol_loss, val_loss = agent.clipped_PPO_loss(mini_batch, minib_old_log_policy, minib_adv, args.clip_eps, writer, args.device)

                optimizer_policy.zero_grad()
                pol_loss.backward()
                optimizer_policy.step()

                optimizer_value.zero_grad()
                val_loss.backward()
                optimizer_value.step()

                pol_loss_acc.append(float(pol_loss))
                val_loss_acc.append(float(val_loss))

        writer.add_scalar('loss/pg_loss', np.mean(pol_loss_acc), n_iter)
        writer.add_scalar('loss/vl_loss', np.mean(val_loss_acc), n_iter)
        writer.add_scalar('data/rew', agent.last_game_rew, n_iter)
        writer.add_scalar('data/mean_rew', np.mean(agent.last_games_rews[-100:]), n_iter)

        if n_iter % args.log_interval == 0:
            test_rews, test_stps = agent.test_game(args.test_episode)
            print(' > Testing..', n_iter, test_rews, test_stps)
            if test_rews > best_test_result:
                torch.save({
                    'agent_policy': args.policy_net.state_dict(),
                    'agent_value': args.value_net.state_dict(),
                    'optimizer_policy': optimizer_policy.state_dict(),
                    'optimizer_value': optimizer_value.state_dict(),
                    'test_reward': test_rews
                }, osp.join(args.log_dir,str(n_iter)+'.pth.tar'))
                best_test_result = test_rews
                print('Best test,  reward:{:.2f}  steps:{}'.format(test_rews, test_stps))

            writer.add_scalar('data/test_rew', test_rews, n_iter)

    writer.close()

