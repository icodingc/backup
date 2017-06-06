import numpy as np
import os,sys
import time
from PIL import Image
from inception.slim import resnet_model as model
try:
    from tensorflow.python.tools import freeze_graph
except ImportError:
    raise ImportError('\n\n\nYou should add freeze_graph.py to tensorflow/python/tools/ manually.\nMaybe also add graph_util.py to tensorflow/python/framework/ \n\n')
import tensorflow as tf
import reader
from scipy import misc

class GenerateGraph():
    """
    @param input model_path
    @return output freeze_graph
    """
    def __init__(self,model_path,graph_name='input_graph'):
        self.x = tf.placeholder(dtype=tf.string)
        img = tf.expand_dims(tf.image.decode_jpeg(self.x,channels=3),0)
        self.gen = model.resnet(tf.cast(img,tf.float32))
        self.out = tf.saturate_cast(self.gen,tf.uint8)
        self.sess = tf.Session()

        assert tf.gfile.Exists(model_path)
        saver = tf.train.Saver()
        print('Using model from {}'.format(model_path))
        saver.restore(self.sess,model_path)

        tf.train.write_graph(self.sess.graph.as_graph_def(),
                    './',graph_name+'_b.pb',False)
        tf.train.write_graph(self.sess.graph.as_graph_def(),
                    './',graph_name+'_text.pb',True)
        print('save graph done...')
    def genArt(self,image_path):
        """
        generate image,compare with freeze_graph
        """
        img_data = tf.gfile.FastGFile(image_path,'r').read()
        out =np.squeeze(self.sess.run(self.out,{self.x:img_data}))
        misc.imsave('out_test_src.png',out)        
        print('save image to out_test_src.png')
def GenerateFreezeGraph(input_graph='input_graph',output_graph='output_graph',model_path=''):
    """
    generate freeze_graph
    """
    print('begin to generate freeze_graph')
    freeze_graph.freeze_graph(input_graph+'_b.pb',"",True,
                model_path,
                'mul','save/restore_all','save/Const:0',
                output_graph+'.pb',True,"")

def TestFreezeGraph(graph_path='output_graph.pb',test_img='test.jpg'):
    with tf.Graph().as_default():
        with tf.gfile.FastGFile(graph_path+'.pb','rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def,name='')
        with tf.Session() as sess:
            img = sess.graph.get_tensor_by_name('Placeholder:0')
            gen = sess.graph.get_tensor_by_name('mul:0')
            img_data = tf.gfile.FastGFile(test_img,'r').read()
            out = np.squeeze(sess.run(gen,{img:img_data}))
            misc.imsave('out_test_graph.png',out)
            print('save image to out_test_graph.png')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--img',dest='img',type=str,default='test.jpg',help='test imge')
    parser.add_argument('--model_path',dest='model_path',type=str,default='./examples/model_style_hb/fast-style-model-9000',help='model path')
    parser.add_argument('--input_g',dest='input_g',type=str,default='input_style_hb',help='input graph name')
    parser.add_argument('--output_g',dest='output_g',type=str,default='output_style_hb',help='output graph name')
    args = parser.parse_args()

    test_img = args.img
    model_path = args.model_path
    input_graph = args.input_g
    output_graph = args.output_g

    T = GenerateGraph(model_path,input_graph)
    T.genArt(test_img)

    GenerateFreezeGraph(input_graph,output_graph,model_path)
    TestFreezeGraph(output_graph,test_img)
