import numpy as np
import os,sys
import time
import tensorflow as tf
from PIL import Image
from inception.slim import resnet_model as model
import reader2 as reader
from scipy import misc

class ArtGenerater():
	"""generate art style with given model"""
	def __init__(self,model_path):
		self.model_path = model_path
		# using string
		self.x = tf.placeholder(dtype=tf.string)
		img = tf.expand_dims(tf.image.decode_jpeg(self.x,channels=3),0)
		self.img = tf.cast(img,tf.float32)
		print self.img
		self.gen = model.resnet(self.img)
		print('--gen')
		self.out = tf.saturate_cast(self.gen,tf.uint8)
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 1.0)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
		file = tf.train.latest_checkpoint(self.model_path)
		if file:
			saver = tf.train.Saver()
			print('Using model from {}'.format(file))
			saver.restore(self.sess,file)
		else:
			print('Could not find trained model')
                tf.train.write_graph(self.sess.graph.as_graph_def(),'./','input_graph_b.pb',False)
                tf.train.write_graph(self.sess.graph.as_graph_def(),'./','input_graph_text.pb',True)
                print "done..."
	def pic(self,image_path):
		img_data = tf.gfile.FastGFile(image_path,'r').read()	
		start = time.time()
		output_t = self.sess.run(self.out,feed_dict={self.x:img_data})
		print('time..',time.time()-start)
		return output_t
if __name__ == '__main__':
	t = ArtGenerater('./examples/model_style_hb')
	pic = t.pic('./test.jpg')
	print pic.shape
	misc.imsave('out.png', pic[0])
	print ('save to out.png')
