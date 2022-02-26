import math, random, copy, time
import numpy as np
import os 
import torch
import torch.nn as nn

import torch.optim as optim
from buffer import PPOMemory
from models import ActorCritic
from config import *
from surviving import Surviving

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:'+cuda_device if USE_CUDA else 'cpu')
current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

class PPO:
    def __init__(self, state_dim, action_dim, hidden_dim, lr, betas, GAMMA, n_epoch, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = GAMMA
        self.eps_clip = eps_clip
        self.n_epoch = n_epoch
        
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
    
    def update(self, memory):   
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            # print(reward,discounted_reward)
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        
        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Optimize policy for K epochs:
        for _ in range(self.n_epoch):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())
                
            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            print(ratios.shape,advantages.shape)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

def main():

    #############################################
    env = Surviving(n_agent = num_agent)
    n_ant = env.n_agent
    state_dim = n_ant*env.len_obs
    action_dim = n_ant*env.n_action

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)
    
    memory = PPOMemory()
    ppo = PPO(state_dim, action_dim, hidden_dim, lr, betas, GAMMA, n_epoch, eps_clip)
    print(lr,betas)
    
    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    
    # 训练日志
    log_file = './log/PPO_log_'+current_time+'.txt'
    # training loop
    for i_episode in range(1, n_episode+1):
        state, adj = env.reset()
        for t in range(max_step):
            timestep += 1
            state = np.array(state)
            # Running policy_old:
            action = ppo.policy_old.act(state, memory)

            # next_obs, next_adj, reward, terminated = env.step(action)
            state, adj, reward, done = env.step(action)
            
            # Saving reward and is_terminal:
            memory.rewards.append(sum(reward))
            memory.is_terminals.append(done)
            
            # update if its time
            if timestep % update_timestep == 0:
                ppo.update(memory)
                memory.clear_memory()
                timestep = 0
            
            running_reward += sum(reward)
            if render:
                env.render()
            if done:
                break
                
        avg_length += t
        
        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}_{}.pth'.format(env_name,lr))
            break
            
        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = running_reward/log_interval
            
            print('Episode {} \t avg length: {} \t reward: {}'.format(i_episode, avg_length, running_reward))
            running_reward = 0
            avg_length = 0
            
if __name__ == '__main__':
    main()

