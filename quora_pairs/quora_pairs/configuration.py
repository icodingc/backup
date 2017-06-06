from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

################
#404290 training sample
#2345796 test sample
#120526 unique tokens
#120527 number of words
################
class ModelConfig(object):
	"""docstring for ModelConfig"""
	def __init__(self):
		"""default model params"""
		self.vocab_size = 120527
		self.max_sequence_length=15
		self.batch_size = 200
		#scale used to initialize model variables
		self.initializer_scale = 0.08
		# word embedding size
		self.embedding_size = 50
		self.num_lstm_units = 256 
		self.logits_size = 128
		self.lstm_dropout_keep_prob=0.5

class TrainingConfig(object):
	def __init__(self):

		self.num_examples_per_epoch = 363861
		
		self.optimizer = "Adam";
		self.initial_learning_rate = 0.05
		self.learning_rate_decay_factor = 0.2
		self.num_epochs_per_decay = 10.0

		#
		self.clip_gradients = 5.0
		self.max_checkpoints_to_keep = 5
