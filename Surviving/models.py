import math, random, os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
from torch.distributions import Categorical
from config import *

os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
USE_CUDA = torch.cuda.is_available()
device = torch.device('cuda:'+cuda_device if USE_CUDA else 'cpu')

Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).cuda() if USE_CUDA else autograd.Variable(*args, **kwargs)

class AttModel(nn.Module):
	def __init__(self, din, hidden_dim, dout):
		super(AttModel, self).__init__()
		self.fcv = nn.Linear(din, hidden_dim)
		self.fck = nn.Linear(din, hidden_dim)
		self.fcq = nn.Linear(din, hidden_dim)
		self.fcout = nn.Linear(hidden_dim, dout)

	def forward(self, x, mask):
		# mask是adjacent matrix
		# x.shape: (1,100,128)
		# v,q,k.shape: (1,100,128)
		# att.shape: (1,100,100)
		# out.shape: (1,100,128)

		v = F.relu(self.fcv(x))
		q = F.relu(self.fcq(x))
		k = F.relu(self.fck(x)).permute(0,2,1)
		att = F.softmax(torch.mul(torch.bmm(q,k), mask) - 9e15*(1 - mask),dim=2)

		out = torch.bmm(att,v)
		# out = torch.add(out,v)
		out = F.relu(self.fcout(out))
		return out

class DGN(nn.Module):
	def __init__(self, n_agent, num_inputs, hidden_dim, num_actions):
		super(DGN, self).__init__()
		
		self.encoder = nn.Linear(num_inputs, hidden_dim)
		self.att_1 = AttModel(hidden_dim, hidden_dim, hidden_dim)
		self.att_2 = AttModel(hidden_dim, hidden_dim, hidden_dim)
		self.q_net = nn.Linear(hidden_dim, num_actions)
		
	def forward(self, x, mask):
		# q.shape: (1,100,5)
		h1 = F.relu(self.encoder(x))
		h2 = self.att_1(h1, mask)
		h3 = self.att_2(h2, mask)
		q = self.q_net(h3)
		return q 

class ActorCritic(nn.Module):
	def __init__(self, state_dim=29, action_dim=5, hidden_dim=128):
		super(ActorCritic, self).__init__()
		self.fc1 = nn.Linear(state_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, action_dim)
		
		self.fc4 = nn.Linear(state_dim, hidden_dim)
		self.fc5 = nn.Linear(hidden_dim, hidden_dim)
		self.fc6 = nn.Linear(hidden_dim, 1)
		self.softmax = nn.Softmax(dim=-1)

	# actor
	def action_layer(self, x):
		# input x.shape: (1,100,29)
		# output x.shape: (1,100,5)
		x = x.view(-1,100,29)
		x = torch.tanh(self.fc1(x))
		x = torch.tanh(self.fc2(x))
		x = torch.tanh(self.fc3(x))
		x = self.softmax(x)
		return x
        
    # critic
	def value_layer(self, x):
		x = x.view(-1,100,29)
		x = torch.tanh(self.fc4(x))
		x = torch.tanh(self.fc5(x))
		x = self.fc6(x)
		print(x.shape)
		return x
        
	def forward(self):
		raise NotImplementedError
        
	def act(self, state, memory):
		state = torch.from_numpy(state).float().to(device) 
		action_probs = self.action_layer(state)
		dist = Categorical(action_probs)
		action = dist.sample()
		
		memory.states.append(state)
		memory.actions.append(action[0])
		# print('log prob,',dist.log_prob(action)[0].sum())
		memory.logprobs.append(dist.log_prob(action)[0].sum())
		
		return action[0].cpu().numpy()# .item()

	def evaluate(self, state, action):
		# action.shape:          (1000, 100), 1000是buffer length
		# action_probs.shape:    (1000, 100, 5)
		# action_logprobs.shape: (1000, 100)
		# dist_entropy.shape:    (1000)
		# state_value.shape:     (1000, 1)
		# torch.squeeze(state_value).shape: (1000)
		action_probs = self.action_layer(state)
		dist = Categorical(action_probs)
		
		action_logprobs = dist.log_prob(action)
		
		dist_entropy = dist.entropy()
		state_value = self.value_layer(state)
		
		return action_logprobs.sum(-1), torch.squeeze(state_value), dist_entropy.sum(-1)
