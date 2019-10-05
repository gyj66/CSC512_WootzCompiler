# CSC512 2018 Fall Course Project: Wootz Compiler 
This project is to develop a compiler called Wootz, which automatically generates TensorFlow code from a Prototxt file. 

## Example Files 
* Prototxt file: inception_v1.prototxt
* Tensorflow code: inception_v1_simple.py
* Multiplexing code: inception_v1_multiplexing.py

## Other Files
* load_graph.py: generate tensorflow log file to ```./summaries``` for tensorboard visualization. To visualize in tensorboard:

``` tensorboard --logdir=./summaries ```



## Notes
### Prototxt to Tensorflow
There are several tools available to convert caffe models to tensorflow. Here are three popular tools:
* [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow)
* [MMdnn](https://github.com/Microsoft/MMdnn)
* [nntools](https://github.com/hahnyuan/nn_tools)

These tools can convert an existing well-trained Caffe model to Tensorflow Flow. For example, the output of caffe-tensorflow 
consists of a python class that constructs the model's graph and a data file containing the model's learned parameters.  
Our project only needs to consider network architecture defined in the prototxt. The values of model paramters (.caffemodel) can be ignored. 

### Mapping from Caffe Layers to Tensorflow Layers
Not all the functionalities specified in Caffe Prototxt can be mapped to the tensorflow operators. For example, Caffe support arbitrary padding whereas tensorflow uses two padding types:```SAME``` and ``` VALID```. In the example tensorflow code ```inception_v1.py```, the padding is set to ```SAME``` as default because it is the classic way to go. 

## References
* The example tensorflow code uses the Tensorflow Slim API: [tensorflow.contrib.slim](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim)
* Tensorflow Model Zoo using Tensorflow Slim API: [Tensorflow Slim Model Library](https://github.com/tensorflow/models/tree/master/research/slim/nets). 
* Tools to visualize the network defined in Caffe Prototxt: [Netscope](https://ethereon.github.io/netscope/#/editor)  