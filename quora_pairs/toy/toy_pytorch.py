#coding=utf-8
import os

# import visdom
import numpy as np

import torch.nn as nn
import torch
from torch import autograd
from torch.autograd import Variable
from torch.optim import RMSprop

import reader

os.environ["CUDA_VISIBLE_DEVICES"]="0"

class TwoLayerNet(nn.Module):
	def __init__(self,D_in,H,D_out):
		super(TwoLayerNet,self).__init__()
		self.linear1 = nn.Linear(D_in,H)
		self.linear2 = nn.Linear(H,D_out)
		self.softmax = nn.LogSoftmax()
	def forward(self,x):
		h_relu = self.linear1(x).clamp(min=0)
		y_pred = self.linear2(h_relu)
		y_prob = self.softmax(y_pred)
		return y_prob

model = TwoLayerNet(23,10,2)
loss_fn = nn.NLLLoss()
optim = RMSprop(model.parameters())

reader = reader.Data(32)
num_steps_per_epoch = len(reader)/32
max_steps = num_steps_per_epoch*5

# viz = visdom.Visdom()


for step in xrange(max_steps):
	step_data = reader.next()

	y_pred = model(Variable(step_data[0]))
	loss = loss_fn(y_pred,Variable(step_data[1]))
	optim.zero_grad()
	loss.backward()
	optim.step()
	if step%20==0:
		print(step,loss.data[0])

