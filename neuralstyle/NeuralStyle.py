import numpy as np
import os,sys
import time
import tensorflow as tf
from PIL import Image
from inception.slim import resnet_model as model
from scipy import misc
from progress.bar import Bar

class ArtGenerater1():
    """generate art style with given model"""
    def __init__(self,model_path):
        self.model_path = model_path
        # using string
        self.x = tf.placeholder(dtype=tf.string)
        img = tf.expand_dims(tf.image.decode_jpeg(self.x,channels=3),0)
        self.img = tf.cast(img,tf.float32)
        self.gen = model.resnet(self.img)
        self.out = tf.saturate_cast(self.gen,tf.uint8)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        assert tf.gfile.Exists(self.model_path)
        saver = tf.train.Saver()
        print('Using model from {}'.format(self.model_path))
        saver.restore(self.sess,self.model_path)

    def pic(self,image_path):
        img_data = tf.gfile.FastGFile(image_path,'r').read()	
        output_t = np.squeeze(self.sess.run(self.out,feed_dict={self.x:img_data}))
        return output_t
class ArtGenerater2():
    """generate art style using freeze_graph"""
    def __init__(self,graph_path):
        self.graph_path = graph_path
        with tf.gfile.FastGFile(self.graph_path,'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def,name='')
        self.sess = tf.Session()
        self.img = self.sess.graph.get_tensor_by_name('Placeholder:0')
        self.gen = self.sess.graph.get_tensor_by_name('mul:0')
    def pic(self,image_path):
        img_data = tf.gfile.FastGFile(image_path,'r').read()
        output_t = np.squeeze(self.sess.run(self.gen,{self.img:img_data}))
        return output_t

if __name__ == '__main__':
#    t1 = ArtGenerater1('./examples/model_style_hb/fast-style-model-9000')
#    pic = t1.pic('./test.jpg')
#    misc.imsave('out.png', pic)
    t2 = ArtGenerater2('./output_graph.pb')
    bar = Bar('Processing',fill='@',max=121)
    start = time.time()
    for f in os.listdir('./try_avi/images_in/'):
        pic1 = t2.pic('./try_avi/images_in/'+f)
        misc.imsave('./try_avi/images_out/'+f, pic1)
        bar.next()
    print time.time()-start
    bar.finish()
