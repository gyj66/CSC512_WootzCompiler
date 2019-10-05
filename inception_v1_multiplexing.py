"""
multiplexing-version of inception_v1 

'multiplexing' means that one can generate different network structures
by calling the function with different arguments. 

The script provides an example of an multiplexing inception v1. 
The number of filters in a component and the component's input can be controlled 
by an extra argument:
* -- config: a dictionary mapping component name to its filter number and input configuration. 

For Inception networks, we only change the filter configurations in an Inception module
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim

def inception_v1(inputs,
                 num_classes=1000,
                 is_training=True,
                 reuse=None,
                 scope='InceptionV1',
                 config=None):

  ############## template code added for multiplexing ##############
  # calculate the number of filter in a conv given config 
  selectdepth = lambda k,v: int(config[k]['ratio']*v) if config and k in config and 'ratio' in config[k] else v 

  # select the input tensor to a module 
  selectinput = lambda k, v: config[k]['input'] if config and k in config and 'input' in config[k] else v 
  ############## end template code ##############

  with tf.variable_scope(scope, "Model", reuse=reuse):
    with slim.arg_scope(default_arg_scope(is_training)):
      
      end_points = {}

      end_point = 'Conv2d_1a_7x7'
      net = slim.conv2d(inputs, 64, [7, 7], stride=2, scope=end_point)
      end_points[end_point] = net


      end_point = 'MaxPool_2a_3x3'
      net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net

      end_point = 'Conv2d_2b_1x1'
      net = slim.conv2d(net, 64, [1, 1], scope=end_point)
      end_points[end_point] = net

      end_point = 'Conv2d_2c_3x3'
      net = slim.conv2d(net, 192, [3, 3], scope=end_point)
      end_points[end_point] = net

      end_point = 'MaxPool_3a_3x3'
      net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net

      end_point = 'Mixed_3b'

      ############## template code added for multiplexing ##############
      net = selectinput(end_point, net)
      ############## end template code ##############

      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 64, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):

          ############## code changes for multiplexing ##############
          # The number of filters (argument name: num_outputs) is 96 in the original model . 
          # In the multiplexing code, the value can be reconfigured by the config argument. 

          # branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(net, selectdepth(end_point,96), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_1 = slim.conv2d(branch_1, 128, [3, 3], scope='Conv2d_0b_3x3')

        with tf.variable_scope('Branch_2'):

          ############## code changes for multiplexing ##############
          # branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(net, selectdepth(end_point,16), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes  ##############

          branch_2 = slim.conv2d(branch_2, 32, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 32, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net


      end_point = 'Mixed_3c'
      ############## template code added for multiplexing ##############
      net = selectinput(end_point, net)
      ############## end template code ##############

      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):

          ############## code changes for multiplexing ##############
          # branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(net, selectdepth(end_point,128), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_1 = slim.conv2d(branch_1, 192, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):

          ############## code changes for multiplexing ##############
          # branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(net, selectdepth(end_point, 32), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_2 = slim.conv2d(branch_2, 96, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net


      end_point = 'MaxPool_4a_3x3'
      net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net


      end_point = 'Mixed_4b'
      ############## template code added for multiplexing ##############
      net = selectinput(end_point, net)
      ############## end template code ##############

      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):

          ############## code changes for multiplexing ##############
          # branch_1 = slim.conv2d(net, 96, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(net, selectdepth(end_point,96), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_1 = slim.conv2d(branch_1, 208, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):

          ############## code changes for multiplexing ##############
          # branch_2 = slim.conv2d(net, 16, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(net, selectdepth(end_point, 16), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_2 = slim.conv2d(branch_2, 48, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net


      end_point = 'Mixed_4c'
      ############## template code added for multiplexing ##############
      net = selectinput(end_point, net)
      ############## end template code ##############
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):

          ############## code changes for multiplexing ##############
          # branch_1 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(net, selectdepth(end_point,112), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_1 = slim.conv2d(branch_1, 224, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):

          ############## code changes for multiplexing ##############
          # branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(net, selectdepth(end_point,24), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net


      end_point = 'Mixed_4d'
      ############## template code added for multiplexing ##############
      net = selectinput(end_point, net)
      ############## end template code ##############
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):

          ############## code changes for multiplexing ##############
          # branch_1 = slim.conv2d(net, 128, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(net, selectdepth(end_point,128), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_1 = slim.conv2d(branch_1, 256, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):

          ############## code changes for multiplexing ##############
          # branch_2 = slim.conv2d(net, 24, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(net, selectdepth(end_point,24), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net


      end_point = 'Mixed_4e'
      ############## template code added for multiplexing ##############
      net = selectinput(end_point, net)
      ############## end template code ##############
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 112, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):

          ############## code changes for multiplexing ##############
          # branch_1 = slim.conv2d(net, 144, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(net, selectdepth(end_point,144), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_1 = slim.conv2d(branch_1, 288, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):

          ############## code changes for multiplexing ##############
          # branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(net, selectdepth(end_point,32), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_2 = slim.conv2d(branch_2, 64, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 64, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net


      end_point = 'Mixed_4f'
      ############## template code added for multiplexing ##############
      net = selectinput(end_point, net)
      ############## end template code ##############
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):

          ############## code changes for multiplexing ##############
          # branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(net, selectdepth(end_point,160), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):

          ############## code changes for multiplexing ##############
          # branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(net, selectdepth(end_point,32), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net


      end_point = 'MaxPool_5a_2x2'
      net = slim.max_pool2d(net, [2, 2], stride=2, scope=end_point)
      end_points[end_point] = net


      end_point = 'Mixed_5b'
      ############## template code added for multiplexing ##############
      net = selectinput(end_point, net)
      ############## end template code ##############
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 256, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):

          ############## code changes for multiplexing ##############
          # branch_1 = slim.conv2d(net, 160, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(net, selectdepth(end_point,160), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_1 = slim.conv2d(branch_1, 320, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):

          ############## code changes for multiplexing ##############
          # branch_2 = slim.conv2d(net, 32, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(net, selectdepth(end_point,32), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0a_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net


      end_point = 'Mixed_5c'
      ############## template code added for multiplexing ##############
      net = selectinput(end_point, net)
      ############## end template code ##############
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):

          ############## code changes for multiplexing ##############
          # branch_1 = slim.conv2d(net, 192, [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(net, selectinput(end_point,192), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_1 = slim.conv2d(branch_1, 384, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):

          ############## code changes for multiplexing ##############
          # branch_2 = slim.conv2d(net, 48, [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(net, selectinput(end_point,48), [1, 1], scope='Conv2d_0a_1x1')
          ############## end code changes##############

          branch_2 = slim.conv2d(branch_2, 128, [3, 3], scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.max_pool2d(net, [3, 3], scope='MaxPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, 128, [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(
            axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      
        
      # Pooling with a fixed kernel size.
      end_point = 'AvgPool_0a_7x7'
      net = slim.avg_pool2d(net, [7, 7], stride=1, scope=end_point)
      end_points[end_point] = net
        
      end_point = 'Logits'
      with tf.variable_scope(end_point):
        net = slim.dropout(net, 0.8, scope='Dropout_0b')
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_0c_1x1')

        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        end_points[end_point] = logits

      end_points['Predictions'] = slim.softmax(logits, scope='Predictions')
  return logits, end_points
inception_v1.default_image_size = 224


def default_arg_scope(is_training=True, 
                        weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):

  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': batch_norm_updates_collections,
      # use fused batch norm if possible.
      'fused': None,
  }
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}

  # Set training state 
  with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
    # Set weight_decay for weights in Conv and FC layers.
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_regularizer=slim.l2_regularizer(weight_decay)):
      # Set batch norm 
      with slim.arg_scope(
          [slim.conv2d],
          normalizer_fn=normalizer_fn,
          normalizer_params=normalizer_params):
          # Set default padding and stride
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                      stride=1, padding='SAME') as sc:
              return sc