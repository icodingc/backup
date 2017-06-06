# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
import utils


batch_norm_params = {
    'decay': 0.9997,
    'epsilon': 0.001,
    'updates_collections': tf.GraphKeys.UPDATE_OPS}

class SimModel(object):
  def __init__(self,config,mode,loss="fc"):
    """Basic setup.
    config: configuration parameters.
    mode:"train","eval" or "inference".
    """
    assert mode in ["train","eval","inference"]
    assert loss in ["pddm","fc"]
    self.config = config
    self.mode = mode
    self.loss = loss

    self.initializer = tf.random_uniform_initializer(
            minval=-self.config.initializer_scale,
            maxval=self.config.initializer_scale)
    # An int32 Tensor with shape [batch_size, padded_length].
    self.input_seqs = None

    # An int32 0/1Tensor with shape [batch_size]
    self.input_mask = None
    # An int32 Tensor with shape [batch_size/2]
    self.labels = None
    # A float32 Tensor with shape [batch_size,padded_length,embedding_size]
    self.embedding_matrix = None#'data/embedding_map.npy'
    self.seq_embeddings = None

    self.features = []
    self.logits = None
    self.preds = None
    self.acc = None
    # A float32 scalar Tensor for the trainer to optimize.
    self.total_loss = None

  def is_training(self):
    return self.mode == "train"

  def build_inputs(self):
    SP = self.config.batch_size
    self.input_seqs = tf.placeholder(
            shape=[SP*2,self.config.max_sequence_length],
            name="input_seqs",
            dtype=tf.int32)
    self.input_mask = tf.placeholder(
            shape=[SP*2,self.config.max_sequence_length],
            name="input_masks",
            dtype=tf.int32)
    # self.feat = tf.placeholder(shape=[SP,23],dtype=tf.float32)
    self.labels = tf.placeholder(shape=[SP],dtype=tf.float32)

  def build_seq_embedding(self):
    """
    Input: 
        self.input_seqs     [batch * pad_length]
    Output: 
        self.seq_embeddings [batch * pad_length * embedding_size]
    """
    with tf.variable_scope("seq_embedding"),tf.device("/cpu:0"):
      if self.embedding_matrix==None:
        initializer=self.initializer
      else:
        initializer=tf.constant_initializer(np.load(self.embedding_matrix))

      embedding_map = tf.get_variable(
                name="map",
                shape=[self.config.vocab_size,self.config.embedding_size],
                initializer=initializer,
                trainable=True)
      # embedding_pad = tf.zeros(
      #           name="pad",
      #           shape=[1,self.config.embedding_size])
      # embedding = tf.concat([embedding_pad,embedding_map],axis=0)
      seq_embeddings = tf.nn.embedding_lookup(
                        embedding_map,
                        self.input_seqs)
      self.seq_embeddings=seq_embeddings

  def build_rnn_feature(self):
    """
    Inputs:
        self.seq_embeddings [batch * length * embedding_size]
    Outputs:
        self.features [batch * logits_size]
    """
    lstm_cell_fw = tf.contrib.rnn.BasicLSTMCell(
            num_units=self.config.num_lstm_units,
            state_is_tuple=True)
    with tf.variable_scope('LSTM',initializer=self.initializer) as lstm_scope:
      #lstm_scope.reuse_variables()
      # Run the batch of sequence embeddigns through the lstm
      # and for mask
      sequence_length = tf.reduce_sum(self.input_mask,1)

      #[batch * padded_length * embedding_size]  bidirectional_
      lstm_outputs,_ = tf.nn.dynamic_rnn(
                             cell=lstm_cell_fw,
                             inputs=self.seq_embeddings,
                             sequence_length=sequence_length,
                             dtype=tf.float32,
                             scope=lstm_scope)
      #mask lstm_outputs
      lstm_outputs = lstm_outputs * tf.expand_dims(tf.to_float(self.input_mask),2)
      lstm_outputs = tf.concat(tf.reduce_max(lstm_outputs,axis=1),axis=1)
      #CNN encoding
    self.features.append(lstm_outputs)

  def build_cnn_feature(self):
    """
    Inputs:
        self.seq_embeddings [batch * length * embedding_size]
    Outputs:
        self.features [batch * logits_size]
    """
    #[batch,sequence_length,embedding_size,1]
    with tf.name_scope("input_text"):
        text_inputs = tf.expand_dims(self.seq_embeddings,-1)
    filter_sizes = [4]
    filter_width = 5
    num_filter_size=64
    with tf.variable_scope("cnn"):
      with slim.arg_scope([slim.conv2d], 
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.05),
                        biases_initializer=tf.zeros_initializer()
                        ):
        with slim.arg_scope([slim.batch_norm],
                            is_training=self.is_training()):
          pooled_outputs = []
          for i,filter_size in enumerate(filter_sizes):
            #conv2d [batch * 15 * 50 * 1]
            net = slim.conv2d(text_inputs,64,
                    kernel_size=[filter_size,filter_width],
                    stride=(1,3),
                    padding="VALID",scope="conv1_"+str(i))
            #avg pooling [batch * 12 * 16 * 64]
            net = slim.avg_pool2d(net,
                    kernel_size=[2,2],
                    stride=2,
                    padding="VALID",scope="pool1_"+str(i))
            #[batch * 6 * 8 * 64]
            net = slim.conv2d(net,128,kernel_size=[1,1],stride=1,
                    padding='VALID',scope="conv2_"+str(i))
            #[batch * 6* 8* 64]
            net = slim.avg_pool2d(net,kernel_size=[2,2],stride=2,
                    padding="VALID",scope="pool2_"+str(i))
            #[batch * 3* 4*128]
            net = slim.avg_pool2d(net,[3,4],scope="pool3_"+str(i))
            pooled_outputs.append(net)
            #[batch * 1 * 1 * (num_filter*2)] features
            merge_pool = tf.concat(pooled_outputs,axis=3)
      self.features.append(tf.nn.l2_normalize(tf.squeeze(merge_pool),1))

  def build_match(self,name=None):
    """
    split feature and forward
    Input:
        self.features  [batch_size * logits_size]
    Output:
        self.logits    [batch_size * 2]
    """
    with tf.variable_scope("MatchLayer"):
      features = tf.concat(self.features,1)
      feat_a,feat_b = tf.split(features,2,axis=0)
      ##############
      # Match layer
      ##############
      # with tf.variable_scope("match"):
      #   M = tf.get_variable(name="M",
      #           shape=[feat_b.get_shape()[1],feat_b.get_shape()[1]],
      #           initializer=self.initializer)
      #   sim_score = utils._cosine_distance(tf.matmul(feat_a,M),feat_b)
      # with tf.variable_scope("cosine"):
      #   cos_score = utils._cosine_distance(feat_a,feat_b)
      ###############
      # or concat other feature. [...]
      ###############
      # vv = tf.square(tf.abs(feat_a-feat_b))
      # uu = tf.multiply(feat_a,feat_b)
      feat_lhs = tf.concat([feat_a,feat_b],axis=1)
      feat_rhs = tf.concat([feat_b,feat_a],axis=1)
      dense = tf.concat([feat_lhs,feat_rhs],axis=0)
         
      with tf.variable_scope("dense_fc"):
        with slim.arg_scope([slim.fully_connected], 
                        normalizer_fn=slim.batch_norm,
                        activation_fn=tf.nn.relu,
                        normalizer_params=batch_norm_params,
                        weights_regularizer=slim.l2_regularizer(0.005),
                        biases_initializer=tf.zeros_initializer()):
          with slim.arg_scope([slim.batch_norm],
                            is_training=self.is_training()):
            dense = slim.fully_connected(dense,self.config.logits_size,scope="dense_1")
            dense = slim.fully_connected(dense,2,scope="dense_2")
      self.logits = dense

  def build_loss(self,name=None):
    """
    Input: self.logits [batch * 2]
    """
    with tf.name_scope(name,"cls_loss",[self.logits,self.labels]):
      labels = tf.to_int32(self.labels)
      logits = self.logits 
      
      logits_up,logits_down = tf.split(logits,2,0)
      logits_all = (logits_up+logits_down)/2.0              #[batch,2]

      predictions = tf.nn.softmax(logits_all)[:,1]

      batch_loss = tf.losses.log_loss(labels,
                    predictions,scope="batch_loss")
      
      l2_loss = tf.add_n(tf.losses.get_regularization_losses())
      total_loss = batch_loss + l2_loss

      preds = tf.to_int32(tf.round(predictions))
      correct_prediction = tf.equal(preds, labels)
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

      tf.summary.scalar("loss/batch_loss",batch_loss)
      tf.summary.scalar("loss/total_loss",total_loss)
      tf.summary.scalar("acc",accuracy) 

      self.preds = predictions
      self.acc = accuracy
      self.total_loss = total_loss

  def setup_global_step(self):
    global_step = tf.Variable(
            initial_value=0,
            name="global_step",
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    self.global_step = global_step

  def build(self):
    """creates all ops for training"""
    self.build_inputs()
    self.build_seq_embedding()
    self.build_rnn_feature()
    self.build_match()
    self.build_loss()
    self.setup_global_step()
