import logging
import math
import os
import random

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import spaces
from gym.utils import seeding
from torch.autograd import Variable
from CartPole import CartPoleEnv
logger = logging.getLogger(__name__)


# CartPole Task Parameter
L_MIN = 0.5   # min length
L_MAX = 5    # max length

# Hyper Parameters
TASK_NUMS = 10
STATE_DIM = 4
ACTION_DIM = 2
TASK_CONFIG_DIM = 3
EPISODE = 1000
STEP = 500
SAMPLE_NUMS = 30 #5,10,20
TEST_SAMPLE_NUMS = 5
ENV_NAME = "CartPole-v0"

# device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ActorNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,action_size):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,action_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = F.log_softmax(self.fc3(out))
        return out

class MetaValueNetwork(nn.Module):

    def __init__(self,input_size,hidden_size,output_size):
        super(MetaValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,hidden_size)
        self.fc3 = nn.Linear(hidden_size,output_size)

    def forward(self,x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class TaskConfigNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(TaskConfigNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Set initial states
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(DEVICE)
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(DEVICE)
        # Forward propagate RNN
        out, _ = self.lstm(x, (h0, c0))
        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])
        return out

def roll_out(actor_network,task,sample_nums):
    states = []
    actions = []
    rewards = []
    is_done = False
    state = task.reset()
    result = 0
    for j in range(sample_nums):
        states.append(state)
        log_softmax_action = actor_network(Variable(torch.Tensor([state])).to(DEVICE))
        softmax_action = torch.exp(log_softmax_action)
        action = np.random.choice(ACTION_DIM,p=softmax_action.cpu().data.numpy()[0])
        one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        next_state,reward,done,_ = task.step(action)
        #task.result += reward
        fix_reward = -10 if done else 1
        actions.append(one_hot_action)
        rewards.append(fix_reward)
        final_state = next_state
        state = next_state
        '''
        if task.result >= 250:
            task.reset()
            break
        '''
        if done:
            is_done = True 
            task.reset()
            #print("result:",result)
            break


    return torch.Tensor(states),torch.Tensor(actions),rewards,is_done,torch.Tensor(final_state)

def discount_reward(r, gamma,final_r):
    discounted_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


def main():
    # Define dimensions of the networks
    
    meta_value_input_dim =  STATE_DIM + TASK_CONFIG_DIM # 7
    task_config_input_dim = STATE_DIM + ACTION_DIM + 1 # 7

    # init meta value network with a task config network
    meta_value_network = MetaValueNetwork(input_size = meta_value_input_dim,hidden_size = 80,output_size = 1)
    task_config_network = TaskConfigNetwork(input_size = task_config_input_dim,hidden_size = 30,num_layers = 1,output_size = 3)
    meta_value_network.to(DEVICE)
    task_config_network.to(DEVICE)

    if os.path.exists("meta_value_network_cartpole.pkl"):
        meta_value_network.load_state_dict(torch.load("meta_value_network_cartpole.pkl"))
        print("load meta value network success")
    if os.path.exists("task_config_network_cartpole.pkl"):
        task_config_network.load_state_dict(torch.load("task_config_network_cartpole.pkl"))
        print("load task config network success")

    meta_value_network_optim = torch.optim.Adam(meta_value_network.parameters(),lr=0.001)
    task_config_network_optim = torch.optim.Adam(task_config_network.parameters(),lr=0.001)

    # init a task generator for data fetching
    task_list = [CartPoleEnv(np.random.uniform(L_MIN, L_MAX))
                 for task in range(TASK_NUMS)]
    [task.reset() for task in task_list]


    for episode in range(EPISODE):
        # ----------------- Training ------------------

        if (episode+1) % 10 ==0 :
            # renew the tasks
            task_list = [CartPoleEnv(np.random.uniform(L_MIN, L_MAX))
                         for task in range(TASK_NUMS)]
            [task.reset() for task in task_list]

        # fetch pre data samples for task config network
        # [task_nums,sample_nums,x+y`]
        
        actor_network_list = [ActorNetwork(STATE_DIM,40,ACTION_DIM) for i in range(TASK_NUMS)]
        [actor_network.to(DEVICE) for actor_network in actor_network_list]
        actor_network_optim_list = [torch.optim.Adam(actor_network.parameters(),lr = 0.01) for actor_network in actor_network_list]

        # sample pre state,action,reward for task config
        pre_states = []
        pre_actions = []
        pre_rewards = []
        for i in range(TASK_NUMS):
            states,actions,rewards,_,_ = roll_out(actor_network_list[i],task_list[i],SAMPLE_NUMS)
            pre_states.append(states)
            pre_actions.append(actions)
            pre_rewards.append(rewards)


        for step in range(STEP):

            for i in range(TASK_NUMS):
                # init task config [1, sample_nums,task_config] task_config size=3
                pre_data_samples = torch.cat((pre_states[i][-9:],pre_actions[i][-9:],torch.Tensor(pre_rewards[i])[-9:].unsqueeze(1)),1).unsqueeze(0)
                task_config = task_config_network(Variable(pre_data_samples).to(DEVICE)) # [1,3]

                states,actions,rewards,is_done,final_state = roll_out(actor_network_list[i],task_list[i],SAMPLE_NUMS)
                final_r = 0
                if not is_done:
                    value_inputs = torch.cat((Variable(final_state.unsqueeze(0)).to(DEVICE),task_config.detach()),1)
                    final_r = meta_value_network(value_inputs).cpu().data.numpy()[0]

                # train actor network
                actor_network_optim_list[i].zero_grad()
                states_var = Variable(states).to(DEVICE)
                
                actions_var = Variable(actions).to(DEVICE)
                task_configs = task_config.repeat(1,len(rewards)).view(-1,3)
                log_softmax_actions = actor_network_list[i](states_var)
                vs = meta_value_network(torch.cat((states_var,task_configs.detach()),1)).detach()
                # calculate qs
                qs = Variable(torch.Tensor(discount_reward(rewards,0.99,final_r))).to(DEVICE)

                advantages = qs - vs
                actor_network_loss = - torch.mean(torch.sum(log_softmax_actions*actions_var,1)* advantages) #+ entropy #+ actor_criterion(actor_y_samples,target_y)
                actor_network_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_network_list[i].parameters(),0.5)

                actor_network_optim_list[i].step()

                # train value network

                meta_value_network_optim.zero_grad()

                target_values = qs
                values = meta_value_network(torch.cat((states_var,task_configs),1))
                criterion = nn.MSELoss()
                meta_value_network_loss = criterion(values,target_values)
                meta_value_network_loss.backward()
                torch.nn.utils.clip_grad_norm_(meta_value_network.parameters(),0.5)

                meta_value_network_optim.step()                
                
                # train actor network
                
                pre_states[i] = states
                pre_actions[i] = actions
                pre_rewards[i] = rewards

                if (step + 1) % 100 == 0:
                    result = 0
                    test_task = CartPoleEnv(np.random.uniform(L_MIN, L_MAX))
                    for test_epi in range(10):
                        state = test_task.reset()
                        for test_step in range(200):
                            softmax_action = torch.exp(actor_network_list[i](Variable(torch.Tensor([state])).to(DEVICE)))
                            #print(softmax_action.data)
                            action = np.argmax(softmax_action.cpu().data.numpy()[0])
                            next_state,reward,done,_ = test_task.step(action)
                            result += reward
                            state = next_state
                            if done:
                                break
                    print("episode:",episode,"task:",i,"step:",step+1,"test result:",result/10.0)

        
        if (episode+1) % 10 == 0 :
            # Save meta value network
            torch.save(meta_value_network.state_dict(),"meta_value_network_cartpole.pkl")
            torch.save(task_config_network.state_dict(),"task_config_network_cartpole.pkl")
            print("save networks for episode:",episode)

if __name__ == '__main__':
    main()
