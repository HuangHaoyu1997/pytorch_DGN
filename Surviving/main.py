import math, random, copy, time
import numpy as np
import os 

import torch
import torch.nn as nn
import torch.optim as optim

from models import DGN, ActorCritic
from buffer import ReplayBuffer, PPOMemory
from surviving import Surviving
from config import *

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:'+cuda_device if USE_CUDA else 'cpu')
current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

# 创建环境
env = Surviving(n_agent = num_agent)
n_ant = env.n_agent
observation_space = env.len_obs
n_actions = env.n_action
print('=====================================')
print('number of agents:', n_ant)
print('observation space:', observation_space)
print('action space:', n_actions)
print('cuda:',device)
print('algorithm:',algorithm)
print('=====================================')
if random_seed:
	torch.manual_seed(random_seed)
	env.seed(random_seed)
	np.random.seed(random_seed)


buff = ReplayBuffer(capacity)
model = DGN(n_ant,observation_space,hidden_dim,n_actions)
model_tar = DGN(n_ant,observation_space,hidden_dim,n_actions)
model = model.to(device)
model_tar = model_tar.to(device)

optimizer = optim.Adam(model.parameters(), lr = lr)

O = np.ones((batch_size, n_ant, observation_space))
Next_O = np.ones((batch_size, n_ant, observation_space))
Matrix = np.ones((batch_size, n_ant, n_ant))
Next_Matrix = np.ones((batch_size, n_ant, n_ant))

# 训练日志
log_file = './log/log_'+current_time+'.txt'

while i_episode < n_episode:

	# epsilon-greedy
	if i_episode > start_train:
		epsilon -= 0.0004
		if epsilon < 0.1:
			epsilon = 0.1
	
	i_episode += 1
	steps = 0
	obs, adj = env.reset()

	while steps < max_step:
		steps += 1 
		action = []
		# obs.shape: (100,29), q.shape: (100,5)
		q = model(
					torch.Tensor(np.array([obs])).to(device), 
		            torch.Tensor(adj).to(device)
					)[0]
		
		for i in range(n_ant):
			if np.random.rand() < epsilon:
				a = np.random.randint(n_actions)
			else:
				a = q[i].argmax().item()
			action.append(a)

		next_obs, next_adj, reward, terminated = env.step(action)

		buff.add(np.array(obs),
				action,
				reward,
				np.array(next_obs),
				adj,
				next_adj,
				terminated)

		obs = next_obs
		adj = next_adj
		score += sum(reward)

	if i_episode % log_interval == 0:
		avg_score = score / (log_interval*num_agent)
		print(i_episode, round(avg_score,3), len(buff.buffer))

		with open(log_file,"a") as f:
			f.write(str(i_episode)+','+str(round(avg_score,3))+'\n')

		score = 0

	# 前100个episode不训练
	if i_episode < start_train:
		continue
	
	# start training
	for e in range(n_epoch):
		
		batch = buff.getBatch(batch_size)
		for j in range(batch_size):
			# content in a batch:
			# (obs, action, reward, new_obs, matrix, next_matrix, done)
			sample = batch[j]
			O[j] = sample[0] # obs
			Next_O[j] = sample[3] # new_obs
			Matrix[j] = sample[4] # matrix
			Next_Matrix[j] = sample[5]

		q_values = model(
						torch.Tensor(O).to(device), 
						torch.Tensor(Matrix).to(device)
						)
		target_q_values = model_tar(
									torch.Tensor(Next_O).to(device), 
									torch.Tensor(Next_Matrix).to(device)
									).max(dim = 2)[0]
		target_q_values = np.array(target_q_values.cpu().data)
		expected_q = np.array(q_values.cpu().data)
		
		for j in range(batch_size):
			sample = batch[j]
			for i in range(n_ant):
				expected_q[j][i][sample[1][i]] = sample[2][i] + (1-sample[6])*GAMMA*target_q_values[j][i]
		
		loss = (q_values - torch.Tensor(expected_q).to(device)).pow(2).mean()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	if i_episode % update_interval == 0:
		model_tar.load_state_dict(model.state_dict())