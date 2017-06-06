# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import tensorflow as tf

import configuration
import sim_model

class TestModel(sim_model.SimModel):
	def build_inputs(self):
		self.input_seqs = tf.random_uniform(
			[self.config.batch_size,15],
			minval=0,
			maxval=self.config.vocab_size,
			dtype=tf.int64)
		self.labels = tf.ones([self.config.batch_size/2])
		self.input_length = tf.reduce_sum(self.input_seqs,1)

class SimModelTest(tf.test.TestCase):
	def setUp(self):
		super(SimModelTest,self).setUp()
		self._model_config=configuration.ModelConfig()
	def _countModelParameters(self):
		counter={}
		for v in tf.global_variables():
			name = v.op.name.split("/")[0]
			num_params=v.get_shape().num_elements()
			assert num_params
			counter[name]=counter.get(name,0)+num_params
		return counter
	def _checkModelParameters(self):
	    """Verifies the number of parameters in the model."""
	    param_counts = self._countModelParameters()
	    expected_param_counts = {
	        # vocab_size * embedding_size
	        "seq_embedding": 24100200,
	        # (embedding_size + num_lstm_units + 1) * 4 * num_lstm_units
	        "lstm": 467968,
	        # (num_lstm_units + 1) * logits_size
	        "logits": 32896,
	        "global_step": 1,
	    }
	    self.assertDictEqual(expected_param_counts, param_counts)
	def _checkOutputs(self, expected_shapes, feed_dict=None):
	    """Verifies that the model produces expected outputs.
	    Args:
	      expected_shapes: A dict mapping Tensor or Tensor name to expected output
	        shape.
	      feed_dict: Values of Tensors to feed into Session.run().
	    """
	    fetches = expected_shapes.keys()

	    with self.test_session() as sess:
	    	sess.run(tf.global_variables_initializer())
	    	outputs = sess.run(fetches, feed_dict)

	    for index, output in enumerate(outputs):
	    	tensor = fetches[index]
	    	expected = expected_shapes[tensor]
	    	actual = output.shape
	    	if expected != actual:
				self.fail("Tensor %s has shape %s (expected %s)." %
	                  (tensor, actual, expected))
	def testBuildForTraining(self):
		model = TestModel(self._model_config, mode="train")
		model.build()
		self._checkModelParameters()
		expected_shapes = {
	        # [batch_size, sequence_length]
	        model.input_seqs: (32, 15),
	        # [batch_size, sequence_length]
	        model.input_length: (32,),
	        # [batch_size, sequence_length, embedding_size]
	        model.seq_embeddings: (32, 15, 200),
	        # Scalar
	        model.total_loss: (),
	        # [batch_size * sequence_length]
	        model.features: (32,128)}
		self._checkOutputs(expected_shapes)
if __name__ == "__main__":
  tf.test.main()