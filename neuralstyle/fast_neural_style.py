from scipy import misc
import numpy as np
import os,sys
import time
import tensorflow as tf
import vgg
from tensorflow.python.tools import freeze_graph

tf.app.flags.DEFINE_float("STYLE_WEIGHT", 1e2, "Weight for style features loss")
tf.app.flags.DEFINE_float("TRANSFORM_WEIGHT", 1e2, "Weight for transform loss")
tf.app.flags.DEFINE_float("TV_WEIGHT", 1e-4, "Weight for total variation loss")
tf.app.flags.DEFINE_string("VGG_PATH", "imagenet-vgg-verydeep-19.mat",
        "Path to vgg model weights")
tf.app.flags.DEFINE_string("MODEL_PATH", "examples/model_gogh", "Path to read/write trained models")
tf.app.flags.DEFINE_string("STYLE_IMAGES_PATH", "pass", "path to styled images")
tf.app.flags.DEFINE_string("TRAIN_IMAGES_PATH", "train2014", "Path to training images")
tf.app.flags.DEFINE_string("STYLE_LAYERS", "relu1_1,relu2_2,relu3_3,relu4_3",
        "Which layers to extract style from")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 2, "Number of concurrent images to train on")
tf.app.flags.DEFINE_integer("EPOCHS", 10, "how many epoch to train")

FLAGS = tf.app.flags.FLAGS

from inception.slim import resnet_model as model
import reader2 as reader

def total_variation_loss(layer):
    shape = tf.shape(layer)
    height = shape[1]
    width = shape[2]
    y = tf.slice(layer, [0,0,0,0], tf.pack([-1,height-1,-1,-1])) - tf.slice(layer, [0,1,0,0], [-1,-1,-1,-1])
    x = tf.slice(layer, [0,0,0,0], tf.pack([-1,-1,width-1,-1])) - tf.slice(layer, [0,0,1,0], [-1,-1,-1,-1])
    return tf.nn.l2_loss(x) / tf.to_float(tf.size(x)) + tf.nn.l2_loss(y) / tf.to_float(tf.size(y))

def main(argv=None):
	if not os.path.exists(FLAGS.MODEL_PATH):os.makedirs(FLAGS.MODEL_PATH)
	images_and_styles = reader.image(FLAGS.TRAIN_IMAGES_PATH,FLAGS.STYLE_IMAGES_PATH,FLAGS.EPOCHS)
	style_layers = FLAGS.STYLE_LAYERS
	if style_layers == 'all':
		style_layers = 'relu1_1,relu1_2,relu2_1,relu2_2,relu3_1,relu3_2,relu3_3,relu3_4,relu4_1,relu4_2,relu4_3,relu4_4,relu5_1,relu5_2,relu5_3,relu5_4' 
	style_layers = style_layers.split(',')

	#TODO how to inplement variable size?
	img = tf.placeholder(tf.float32,[1,256,256,3])
	styles = tf.placeholder(tf.float32,[1,256,256,3])
	generated = model.resnet(img)

	style_loss = 0
	if FLAGS.STYLE_WEIGHT > 0.0:
		net, _ = vgg.net(FLAGS.VGG_PATH, tf.concat(0,[generated,styles]))
		for layer in style_layers:
			genefeat,stylefeat = tf.split(0,2,net[layer])
			size = tf.size(genefeat)
			style_loss += tf.nn.l2_loss(genefeat-stylefeat) / tf.to_float(size)
		style_loss = style_loss / len(style_layers)
	else:
		print "only tranfrom loss,just like autoencoder"
        #TODO .. content_loss ?
	transform_loss = tf.nn.l2_loss(generated - styles)/tf.to_float(tf.size(img))
	loss = FLAGS.STYLE_WEIGHT * style_loss + FLAGS.TRANSFORM_WEIGHT * transform_loss + \
		    FLAGS.TV_WEIGHT * total_variation_loss(generated)

	global_step = tf.Variable(0, name="global_step", trainable=False)
	train_op = tf.train.AdamOptimizer(1e-3).minimize(loss, global_step=global_step)
	output_format = tf.saturate_cast(tf.concat(0, [generated, img, styles]), tf.uint8)

	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction=0.4
	with tf.Session(config=config) as sess:
		saver = tf.train.Saver(tf.all_variables())
		file = tf.train.latest_checkpoint(FLAGS.MODEL_PATH)
		if file:
			print('Restoring model from {}'.format(file))
			saver.restore(sess, file)
		else:
			print('New model initilized')
			sess.run(tf.initialize_all_variables())
		for img_,sle_ in images_and_styles:
			img_  = np.expand_dims(img_,0)
			styles_ = np.expand_dims(sle_,0)
			feed_dict={img:img_,styles:styles_}
			_, loss_t, step = sess.run([train_op, loss, global_step],feed_dict=feed_dict)
			if step % 10 == 0:
				print('train in {},loss = {}'.format(step, loss_t))
			if step % 100 == 0:
				output_t = sess.run(output_format,feed_dict)
				misc.imsave('step_{}_out_gen.png'.format(step), output_t[0])
				misc.imsave('step_{}_out_img.png'.format(step), output_t[1])
				misc.imsave('step_{}_out_sty.png'.format(step), output_t[2])
			if step % 10 == 0:
				saver.save(sess, FLAGS.MODEL_PATH + '/fast-style-model', global_step=step)
                                tf.train.write_graph(sess.graph.as_graph_def(),'./','input_graph_text.pb',True)
                                tf.train.write_graph(sess.graph.as_graph_def(),'./','input_graph_b.pb',False)
                                print('saved graph')
if __name__ == '__main__':
    tf.app.run()
