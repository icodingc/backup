import tensorflow as tf
from scipy import misc
with tf.Graph().as_default():
  with tf.gfile.FastGFile('output_graph.pb','rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def,name='')
  with tf.Session() as sess:
    img = sess.graph.get_tensor_by_name("Placeholder:0")
    out_node = sess.graph.get_tensor_by_name("mul:0")
    img_data = tf.gfile.FastGFile('../test.jpg','r').read()
    out = sess.run(out_node,{img:img_data})
    print out.shape
    misc.imsave('out_graph.png', out[0])
    print('saved..')
