from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import resnet_model as resnet


class InceptionTest(tf.test.TestCase):

  def testBuildLogits(self):
    batch_size = 1
    height, width = 512, 512
    num_classes = 10
    with self.test_session():
      inputs = tf.random_uniform((batch_size, height, width, 3))
      logits= resnet.net(inputs)
      self.assertListEqual(logits.get_shape().as_list(),
                           [batch_size, height,width,3])

if __name__ == '__main__':
  tf.test.main()
