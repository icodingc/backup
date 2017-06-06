from scipy import misc
import numpy as np
import os,sys
import time
import tensorflow as tf
import vgg
from PIL import Image

tf.app.flags.DEFINE_string("VGG_PATH", "imagenet-vgg-verydeep-19.mat",
				"Path to vgg model weights")
tf.app.flags.DEFINE_string("MODEL_PATH", "examples/model_style_hb", "Path to read/write trained models")
tf.app.flags.DEFINE_string("STYLE_IMAGES_PATH", "pass", "path to styled images")
tf.app.flags.DEFINE_string("TRAIN_IMAGES_PATH", "train2014", "Path to training images")
tf.app.flags.DEFINE_integer("EPOCHS", 10, "how many epoch to train")

FLAGS = tf.app.flags.FLAGS

from inception.slim import resnet_model as model
import reader

def main(argv=None):
	#images_and_styles = reader.image(FLAGS.TRAIN_IMAGES_PATH,FLAGS.STYLE_IMAGES_PATH,1)
	#TODO how to inplement variable size?
	#img = tf.placeholder(tf.float32,[1,512,512,3])
	img = np.asarray(Image.open('test.jpg')).astype(np.float32)
	img = np.expand_dims(img,0)
	generated = model.resnet(tf.convert_to_tensor(img,dtype=tf.float32))
	output_format = tf.saturate_cast(generated, tf.uint8)

	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction=0.8
    	with tf.Session(config=config) as sess:
		file = tf.train.latest_checkpoint(FLAGS.MODEL_PATH)
		saver = tf.train.Saver()
		if file:
			print('Using model from {}'.format(file))
			saver.restore(sess, file)
		else:
			print('Could not find trained model')
		start = time.time()
		output_t = sess.run(output_format)
		print('time..',time.time()-start)
		for i, raw_image in enumerate(output_t):misc.imsave('out_test{0:04d}.png'.format(i), raw_image)
                #saver = tf.train.Saver()
                #saver.save(sess,'ckpt',global_step=1,latest_filename='ckpt_state')
                tf.train.write_graph(sess.graph.as_graph_def(),'./','input_graph_b.pb',False)
                tf.train.write_graph(sess.graph.as_graph_def(),'./','input_graph_text.pb',True)
                print('done..')
		sys.exit()
		for img_,sle_ in images_and_styles:
			img_	= np.expand_dims(img_,0)
			feed_dict={img:img_}
			output_t = sess.run(output_format,feed_dict)
			for i, raw_image in enumerate(output_t):misc.imsave('out{0:04d}.png'.format(i), raw_image)
if __name__ == '__main__':
		tf.app.run()
