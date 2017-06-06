#coding=utf-8
import numpy as np
from sklearn.utils import shuffle

import torch

datas = np.load('data/feature_sec.npy')
labels = np.load('data/labels.npy')
print("train data",datas.shape)


class Data(object):
	def __init__(self,batch_size=64):
		self._cnt = 0
		self._len = len(datas)
		self.batch_size=batch_size
		self._datas = datas
		self._labels = labels
	def shuffle(self):
		self._datas,self._labels = shuffle(datas,labels) 
	def __len__(self):
		return self._len
	def next(self):
		if self._cnt+self.batch_size>= self._len:
			self._cnt=0
			self.shuffle()
		beg = self._cnt
		end = beg+self.batch_size
		self._cnt=end
		return (torch.FloatTensor(self._datas[beg:end]),
				torch.LongTensor(self._labels[beg:end]))
