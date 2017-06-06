import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from PIL import Image
import numpy as np
print "start.."
freeze_graph.freeze_graph('./input_graph_b.pb',"",True,
                './examples/model_style_hb/fast-style-model-9000',
                'mul',
                'save/restore_all','save/Const:0',
                './output_graph.pb',
                True,"")

print "end..."
