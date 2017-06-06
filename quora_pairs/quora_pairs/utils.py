#coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def _cosine_distance(a,b):
  a.get_shape().assert_is_compatible_with(b.get_shape())
  return tf.expand_dims(tf.reduce_sum(tf.multiply(a,b),1),1)


def _euclidean_distance(a,b):
  a.get_shape().assert_is_compatible_with(b.get_shape())
  return tf.reduce_sum(tf.square(tf.sub(a,b)),1)

#################
##adapted from dennybritz cnn-text-classification-tf
#################

import re
import itertools
from collections import Counter

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
def load_data_and_labels(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r").readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r").readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [clean_str(sent) for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


##################
#useless
##################
  # def build_loss_mse(self,name=None):
  #       """
  #       split feature and compute loss
  #       @1. mse
  #       Input:
  #           self.features  [batch_size * logits_size]
  #           self.labels    [batch_size/2]
  #       Output:
  #           self.total_loss
  #       """
  #       with tf.name_scope(name,"mse_loss",[self.features,self.labels]):
  #           feats = tf.nn.l2_normalize(self.features,1)
  #           feat_a,feat_b = tf.split(feats,2,axis=0)
  #           #print(feat_a,feat_b)  [batch_size,embedding_size]
  #           similarities = tf.squeeze(utils._cosine_distance(feat_a,feat_b)) #(batch/2,)
  #           similarities = tf.sigmoid(similarities)
  #           labels = tf.squeeze(self.labels)
            
  #           batch_loss = tf.losses.log_loss(labels,similarities)
  #           total_loss=tf.losses.get_total_loss()

  #           correct_prediction = tf.equal(tf.to_int32(tf.round(similarities)), labels)
  #           accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  #       # add summaries.
  #       tf.summary.scalar("losses/batch_loss",batch_loss)
  #       tf.summary.scalar("losses/total_loss",total_loss)
  #       tf.summary.scalar("acc",accuracy)

  #       self.preds = None
  #       self.total_loss = total_loss
  #   def build_loss_siamese(self,name=None):
  #       """
  #       split feature and compute loss
  #       @2. contrastive loss? how to inference ?
  #       Input:
  #           self.features  [batch_size * logits_size]
  #           self.labels    [batch_size/2]
  #       Output:
  #           self.total_loss
  #       """
  #       with tf.name_scope(name,"siamese_loss",[self.features,self.labels]):
  #           feats = tf.nn.l2_normalize(self.features,1)
  #           feat_a,feat_b = tf.split(feats,2,axis=0)

  #           dists = tf.squeeze(utils._euclidean_distance(feat_a,feat_b))
  #           labels = tf.squeeze(self.labels)

  #           siamese_loss = tf.reduce_mean(labels*dists + (1-labels)*tf.nn.relu(1.-dists),0)
            
  #           tf.losses.add_loss(siamese_loss)
  #           total_loss=tf.losses.get_total_loss()
  #           #print(total_loss)
  #       # add summaries.
  #       tf.summary.scalar("losses/total_loss",total_loss)

  #       self.preds = None
  #       self.total_loss = total_loss

      # def build_loss_hybrid(self,name=None):
      #   """
      #   split feature and compute loss
      #   @3. pddm
      #   Input:
      #       self.features  [batch_size * logits_size]
      #       self.labels    [batch_size/2]
      #   Output:
      #       self.total_loss
      #   """
      #   with tf.name_scope(name,"hybrid",[self.features,self.labels]):

      #       feat_a,feat_b = tf.split(tf.nn.l2_normalize(
      #                       self.features,1)
      #                       ,2,axis=0)
      #       vv = tf.abs(feat_a-feat_b)
      #       uu = tf.multiply(feat_a,feat_b)

      #       with tf.variable_scope("hybrid_fc"):
      #           score_abs = slim.linear(vv,1,scope="hybrid_abs")
      #           score_mul = slim.linear(uu,1,scope="hybrid_mul")

      #       scores = tf.sigmoid(score_abs + score_mul)   #[batch_size,]
      #       labels = tf.to_int32(self.labels) #[batch_size,]

      #       batch_loss = tf.losses.log_loss(labels,tf.squeeze(scores))
      #       total_loss = tf.losses.get_total_loss()

      #       correct_prediction = tf.equal(tf.to_int32(tf.round(scores)), labels)
      #       accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      #   # add summaries.
      #   tf.summary.scalar("losses/batch_loss",batch_loss)
      #   tf.summary.scalar("losses/total_loss",total_loss)
      #   tf.summary.scalar("acc",accuracy)

      #   self.preds = None
      #   self.total_loss = total_loss
  # def build_pddm(self,name=None):
  #   """
  #   split feature and pddm unit
  #   Input:
  #       self.features  [batch_size * logits_size]
  #   Output:
  #       self.logits    [batch_size * 2]
  #   """
  #   with tf.variable_scope("PDDM"):
  #     feat_a,feat_b = tf.split(tf.nn.l2_normalize(self.features,1),
  #                             2,axis=0)
  #     feats = tf.concat([feat_a,feat_b],axis=1)
  #     vv = tf.abs(feat_a - feat_b)
  #     uu = (feat_a + feat_b)/2.0
                    
  #     with slim.arg_scope([slim.fully_connected], 
  #                       normalizer_fn=slim.batch_norm,
  #                       activation_fn=tf.nn.relu,
  #                       normalizer_params=batch_norm_params,
  #                       weights_regularizer=slim.l2_regularizer(0.0005),
  #                       biases_initializer=tf.zeros_initializer()):
  #       #TODO should add bn=is_training?
  #       dense0 = slim.fully_connected(feats,2,scope="dense_0")

  #       pddm_abs = slim.fully_connected(vv,self.config.logits_size,scope="pddm_abs")
  #       pddm_add = slim.fully_connected(uu,self.config.logits_size,scope="pddm_add")
  #       pddm_abs = tf.nn.l2_normalize(pddm_abs,1)
  #       pddm_add = tf.nn.l2_normalize(pddm_add,1)
  #       dense = slim.fully_connected(tf.concat([pddm_abs,pddm_add],axis=1),
  #               self.config.logits_size,scope="dense_1")
  #       dense = slim.fully_connected(dense,2,scope="dense_2")


  #     self.logits = dense+dense0



    # def build_cosine(self,name=None):
    # """
    # split feature and forward
    # Input:
    #     self.features  [batch_size * logits_size]
    # Output:
    #     cosine matching
    # """
    # with tf.variable_scope("cosine"):
    #   features = tf.nn.l2_normalize(tf.concat(self.features,1),1)

    #   feat_a,feat_b = tf.split(features,2,axis=0)

    #   sims = tf.clip_by_value(utils._cosine_distance(feat_a,feat_b),0,1)
    #   sims = tf.squeeze(sims)
    #   labels = tf.to_int32(self.labels)
    #   batch_loss = tf.losses.log_loss(labels,
    #       sims,scope="batch_loss")
    #   total_loss = batch_loss

    #   preds = tf.to_int32(tf.round(sims))
    #   correct_prediction = tf.equal(preds, labels)
    #   accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #   tf.summary.scalar("loss/batch_loss",batch_loss)
    #   tf.summary.scalar("loss/total_loss",total_loss)
    #   tf.summary.scalar("acc",accuracy) 

    #   self.preds = sims
    #   self.acc = accuracy
    #   self.total_loss = total_loss