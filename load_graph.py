import os 
import tensorflow as tf 
import tensorflow.contrib.slim as slim 
from  inception_v1_simple import inception_v1
from inception_v1_multiplexing import inception_v1 as inception_v1_mul 

def prepare_file_system(directory):
        # Setup the directory we'll write summaries to for TensorBoard
        if tf.gfile.Exists(directory):
                tf.gfile.DeleteRecursively(directory)
        tf.gfile.MakeDirs(directory)
        return


logdir = './summaries/'
prepare_file_system(logdir)

with tf.Graph().as_default():
	# 1. load the standard inception_v1 graph 
	image_size = inception_v1.default_image_size 
	inputs = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))
	inception_v1(inputs, scope='InceptionV1')



	# 2. load the multiplexing inception_v1 graph with a customized config
	image_size = inception_v1_mul.default_image_size 
	inputs = tf.placeholder(tf.float32, shape=(None, image_size, image_size, 3))
	config={'Mixed_4b': 
				{'ratio': 0.5
				},
		  	'Mixed_4c':
		  		{ 'ratio': 0.5,	
		  		'input': tf.placeholder(tf.float32, shape=(None, 14, 14, 512))
		  		}}
	inception_v1_mul(inputs, config = config, scope='InceptionV1_mul')

	with tf.Session() as sess:
		writer = tf.summary.FileWriter(logdir, sess.graph)

