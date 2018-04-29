import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import time
# intuition: each neuron should focus on different things,
# thus the standard deviation for weights and biases should grow
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)

# convolution with padding = 'SAME'
# out_height = ceil(float(in_height) / float(strides[1]))
# out_width  = ceil(float(in_width) / float(strides[
#
#
# ]))

# default: keep_prob = 1 means no dropout
def conv_layer(filter_side, input_tensor, out_channels, layer_name, mu = 0, sigma = 0.1, act=tf.nn.relu, keep_prob = 1):
    with tf.name_scope(layer_name):
        # shape[3].value gives int as output
        in_channels = input_tensor.shape[3].value
        print("in_channels", in_channels)
        print("shape", filter_side, filter_side, in_channels, out_channels)
        print("type", type(filter_side), type(filter_side), type(in_channels), type(out_channels))
        # define filter
        # shape [filter_height, filter_width, in_channels, out_channels]
        conv_W = tf.Variable(tf.truncated_normal(shape=(filter_side, filter_side, in_channels, out_channels), mean = mu, stddev = sigma))
        print("conv_W", conv_W)

        conv_b = tf.Variable(tf.zeros(out_channels))

        # convolution
        preactivate = tf.nn.conv2d(input_tensor, conv_W, strides=[1,1,1,1], padding='SAME') + conv_b
        print("preactivate", preactivate.shape)
        # activation
        activation = act(preactivate, name = 'activation')
        # max pool
        maxPool = tf.nn.max_pool(activation, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # dropout
        output_tensor = tf.nn.dropout(maxPool, keep_prob)

    return output_tensor


# keep_prob = 1, no dropout
# final fc layer: logitsLayer = True
def fc_layer(input_tensor, output_dim, layer_name, act=tf.nn.relu, keep_prob = 1, logitsLayer = False):
    """Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # input_tensor.shape[0] is number of examples
        input_dim = input_tensor.shape[1].value

        print("nput_dim, output_dim", input_dim, output_dim)

        weights = tf.Variable(tf.truncated_normal(shape=(input_dim, output_dim), stddev=0.1))
        variable_summaries(weights)
        biases = tf.Variable(tf.zeros(output_dim))
        variable_summaries(biases)
        preactivate = tf.matmul(input_tensor, weights) + biases

        # only perform wx + b if it's the final fc layer
        if (logitsLayer == True):
            logits = preactivate
            return logits
        # if it's not a final fc layer, perform activation and dropout
        activation = act(preactivate, name = 'activation')
        if (keep_prob == 1):
            return activation
    with tf.name_scope('dropout'):
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        dropped = tf.nn.dropout(activation, keep_prob)

    return dropped



