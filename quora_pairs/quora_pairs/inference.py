#coding=utf-8
"""inference the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import re
import os
import os.path
from tqdm import tqdm

import numpy as np
import pandas as pd
import tensorflow as tf

import configuration
import sim_model
import test_reader as reader

os.environ["CUDA_VISIBLE_DEVICES"]="0"

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_dir", "./logs/cnn2",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_string("submit_name", "cnn_concat",
                       "submit name")
tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
  assert FLAGS.checkpoint_dir, "--checkpoint_dir is required"

  model_config = configuration.ModelConfig()
  training_config = configuration.TrainingConfig()
  model_config.batch_size=612

  # Build The tf Graph
  g = tf.Graph()
  with g.as_default():
    # Build the model,care about BN and scope-prefix
    with tf.variable_scope("train"):
      model = sim_model.SimModel(
        model_config,mode="inference")
      model.build()

    # Set up the Saver
    restore = tf.train.Saver()

  with g.as_default():
    init = tf.global_variables_initializer()

    # start 
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options,
        allow_soft_placement=True))
    sess.run(init)

    ##Restore 
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
    	restore.restore(sess,ckpt.model_checkpoint_path)
    	print('Successfully loaded model from %s' % ckpt.model_checkpoint_path)
    else:
    	print('No checkpoint file found at %s' % FLAGS.checkpoint_dir)
    	return
    #TODO care about batch_size
    preds_all =[]
    for feats in tqdm(reader.batch_inputs()):
      start_time = time.time()
      feed_dict={
        model.input_seqs:feats[0],
        model.input_mask:feats[1],
        model.labels:feats[2],
      }
      score_value = sess.run(model.preds,feed_dict)
      preds_all.append(np.squeeze(score_value))
    # generate submit
    test_ids = np.load('data/test/test_ids.npy')
    preds = np.hstack(preds_all) # * .75
    assert len(test_ids)==len(preds)
    submission = pd.DataFrame({'is_duplicate':preds.ravel(),'test_id':test_ids})
    submission.to_csv('submit_logs/'+FLAGS.submit_name+'.csv', index=False)
    print("done!")
    
if __name__ == "__main__":
  tf.app.run()
