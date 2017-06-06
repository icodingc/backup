#coding=utf-8
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time
import re
import os
import os.path

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import configuration
import sim_model
import reader

os.environ["CUDA_VISIBLE_DEVICES"]="0"

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("train_dir", "./logs/lstm_adam",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 45000, "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 20,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_string("checkpoint_dir", "./logs/cnn_fc",
                       "Directory for fine-tune")
tf.flags.DEFINE_boolean("debug","False","is debug")

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
  assert FLAGS.train_dir, "--train_dir is required"

  np.random.seed(1)

  model_config = configuration.ModelConfig()
  training_config = configuration.TrainingConfig()

  # Create training directory.
  train_dir = FLAGS.train_dir
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info("Creating training directory: %s", train_dir)
    tf.gfile.MakeDirs(train_dir)

  # Build The tf Graph
  g = tf.Graph()
  with g.as_default():
    # Build the model
    with tf.variable_scope("train"):
      model = sim_model.SimModel(model_config,mode="train")
      model.build()
    #TODO(error ? placeholder)
    # with tf.variable_scope("train",reuse=True):
    #   eval_model = sim_model.SimModel(model_config,mode="eval")
    #   eval_model.build()    

    # Set up the learning rate.
    learning_rate_decay_fn=None
    learning_rate = tf.constant(training_config.initial_learning_rate)
    if training_config.learning_rate_decay_factor > 0:
      num_batches_per_epoch = (training_config.num_examples_per_epoch/
                    model_config.batch_size)
      decay_steps = int(num_batches_per_epoch*
                training_config.num_epochs_per_decay)
      def _learning_rate_decay_fn(learning_rate, global_step):
        return tf.train.exponential_decay(
              learning_rate,
              global_step,
              decay_steps=decay_steps,
              decay_rate=training_config.learning_rate_decay_factor,
              staircase=True)
      learning_rate_decay_fn = _learning_rate_decay_fn

    optimizer = tf.train.AdamOptimizer(
                    learning_rate,
                    beta1=0.9,
                    beta2=0.999,
                    epsilon=1.0) 
    # Set up the training ops.
    train_op = tf.contrib.layers.optimize_loss(
        loss=model.total_loss,
        global_step=model.global_step,
        learning_rate=learning_rate,
        optimizer=optimizer,
        clip_gradients=training_config.clip_gradients,
        learning_rate_decay_fn=learning_rate_decay_fn)
    #################
    # Summary
    #################
    for var in tf.trainable_variables():
      tf.summary.histogram("params/"+var.op.name,var)
    # Set up the Saver
    saver = tf.train.Saver(max_to_keep=training_config.max_checkpoints_to_keep)
    # restore = tf.train.Saver()
  # Run training.
  with g.as_default():
    global_step = model.global_step

    init = tf.global_variables_initializer()

    dev_summary_op = tf.summary.merge_all()
    train_summary_op = dev_summary_op
    ##################
    # session config
    ##################
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.35)
    sess = tf.Session(config=tf.ConfigProto(
        device_count={"CPU": 4}, # limit to num_cpu_core CPU usage  
        intra_op_parallelism_threads=2,
        inter_op_parallelism_threads=2,
        gpu_options=gpu_options,
        allow_soft_placement=True))
    sess.run(init)
    ###################
    # debug
    # https://www.tensorflow.org/programmers_guide/debugger
    # `run -f has_inf_or_nan`
    ###################
    if FLAGS.debug==True:
      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    ###################
    # Restore checkpoint
    ####################
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess,ckpt.model_checkpoint_path)
      print('Successfully loaded model from %s' % ckpt.model_checkpoint_path)
    else:
      print('No checkpoint file found at %s' % FLAGS.checkpoint_dir)

    summary_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.train_dir,"summaries","train"),
        graph=sess.graph)
    dev_summary_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.train_dir,"summaries","dev"),
        graph=sess.graph)
    #TODO should read data
    test_reader = reader.Test_batch()
    step=0
    for feats in reader.batch_inputs():
      step=step+1
      if step > FLAGS.number_of_steps:break

      start_time = time.time()
      feed_dict={
        model.input_seqs:feats[0],
        model.input_mask:feats[1],
        # model.feat:feats[2],
        model.labels:feats[2],
      }

      loss_value,acc_value = sess.run([train_op,model.acc],feed_dict)
      duration = time.time()-start_time

      assert not np.isnan(loss_value),'Model diverged with loss = NaN'

      if step%50==0:
        # examples_per_sec = model_config.batch_size / float(duration)
        format_str = ('%s: step %d, loss = %.2f ,acc = %.2f')
        print(format_str % (datetime.now(), step, np.mean(loss_value),acc_value))

      if step%200 ==0:
        summary_str = sess.run(train_summary_op,feed_dict)
        summary_writer.add_summary(summary_str, step)

      if step%400 ==0:
        dev_data = test_reader.next()
        feed_dict={
        model.input_seqs:dev_data[0],
        model.input_mask:dev_data[1],
        # model.feat:dev_data[2],
        model.labels:dev_data[2],}

        dev_summary_str = sess.run(dev_summary_op,feed_dict)
        dev_summary_writer.add_summary(dev_summary_str, step)

      if step%5000==0 or (step+1)==FLAGS.number_of_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

if __name__ == "__main__":
  tf.app.run()
