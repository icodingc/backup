import tensorflow as tf
slim = tf.contrib.slim

def net(inputs,is_training=True,scope='res'):
  with slim.arg_scope([slim.batch_norm],is_training=is_training):
    net = slim.conv2d(inputs, 32, [9, 9], stride=1,scope='conv1')
    net = slim.conv2d(net, 64, [3, 3],stride=2,scope='conv2')
    net = slim.conv2d(net, 128, [3, 3],stride=2,scope='conv3')
    added = net
    for i in xrange(5):
      x = added
      net = slim.conv2d(x, 128, [3, 3], stride=1,scope='res'+str(i)+'c1')
      net = slim.conv2d(net, 128, [3, 3], activation_fn=None,stride=1,scope='res'+str(i)+'c2')
    added = x + net

    net = slim.conv2d_transpose(added,64,[3,3],stride=2,scope='deconv1')
    net = slim.conv2d_transpose(net,32,[3,3],stride=2,scope='deconv2')
    net = slim.conv2d_transpose(net,3,[9,9],stride=1,activation_fn=tf.nn.tanh,scope='deconv3')
    return (net + 1)*127.5
def resnet(inputs,is_train=True,scope='res'):
  batch_norm_params = {
      	'decay': 0.9997,
      	'epsilon': 0.001,}
  with slim.arg_scope([slim.conv2d,slim.conv2d_transpose],
                                weights_regularizer=slim.l2_regularizer(0.0005),
				activation_fn=tf.nn.relu,
				normalizer_fn =slim.batch_norm,
				normalizer_params=batch_norm_params):
    return net(inputs,is_train,scope)
