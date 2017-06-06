# resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from inception.slim import ops
from inception.slim import scopes


def net(inputs,is_training=True,scope=''):
  with tf.op_scope([inputs], scope, 'resnet'):
    with scopes.arg_scope([ops.conv2d,ops.deconv2d,ops.batch_norm],
                          is_training=is_training):
        # 256 x 256 x 3
        net = ops.conv2d(inputs, 32, [9, 9], stride=1,scope='conv1')
        # 256 x 256 x 32
        net = ops.conv2d(net, 64, [3, 3],stride=2,scope='conv2')
        # 128 x 128 x 64
        net = ops.conv2d(net, 128, [3, 3],stride=2,scope='conv3')
        # 64 x 64 x 128
	added = net
	for i in xrange(5):
		x = added
        	net = ops.conv2d(x, 128, [3, 3], stride=1,scope='res'+str(i)+'c1')
        	net = ops.conv2d(net, 128, [3, 3], activation=None,stride=1,scope='res'+str(i)+'c2')
		added = x + net
	net = added
	# print net
        # 64 x 64 x 128
	net = ops.deconv2d(net,64,[3,3],stride=2,scope='deconv1')
	net = ops.deconv2d(net,32,[3,3],stride=2,scope='deconv2')
	net = ops.deconv2d(net,3,[9,9],stride=1,activation=tf.nn.tanh,scope='deconv3')
        return (net + 1)*127.5
def resnet(inputs,is_train=True,scope=''):
	batch_norm_params = {
      	'decay': 0.9997,
      	'epsilon': 0.001,}
	with scopes.arg_scope([ops.conv2d,ops.deconv2d],weight_decay=0.0005,
				stddev=0.1,
				activation=tf.nn.relu,
				batch_norm_params=batch_norm_params):
		return net(inputs,is_train,scope)
