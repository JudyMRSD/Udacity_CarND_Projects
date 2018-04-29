# Lenet use gpu
# https://github.com/tiagofrepereira2012/examples.tensorflow/blob/master/examples/tensorflow/lenet.py
# VAE
# https://github.com/allenovo/conditional_vae/blob/master/vae.py
# MNIST
# https://github.com/tensorflow/tensorflow/blob/r1.1/tensorflow/examples/tutorials/mnist/mnist_softmax.py
# Tensorboard:
# http://blog.csdn.net/sinat_33761963/article/details/62433234

# group layers
# https://github.com/tiagofrepereira2012/examples.tensorflow/blob/master/examples/tensorflow/lenet.py
#

# conv3 = conv_layer(filter_side=conv3_kernel_size, input_tensor=conv2, out_channels=conv3_output_channel, layer_name='conv3')
# fc0 = flatten(conv2)
import tensorflow as tf
from tfUtil import *
from tensorflow.contrib.layers import flatten


class LeNet():
    def __init__(self,
                 img_w = 32,
                 img_h = 32,
                 img_channel = 1,
                 conv1_kernel_size = 5,
                 conv1_output_channel = 6,

                 conv2_kernel_size = 5,
                 conv2_output_channel = 16,

                 conv3_kernel_size=5,
                 conv3_output_channel=16,

                 fc1_out = 120,
                 fc2_out = 84,

                 n_classes = 10):

        with tf.name_scope('LeNet'):
            with tf.name_scope('input'):
                self.x = tf.placeholder(tf.float32, [None, img_w, img_h, img_channel], name='input')
                self.labels = tf.placeholder(tf.float32, [None], name='output')
                one_hot_labels = tf.one_hot(indices=tf.cast(self.labels, tf.int32), depth=n_classes)
            # convolution, relu:  conv1 (?, 32, 32, 6)
            # max pool:  conv1 (?, 16, 16, 6)
            conv1 = conv_layer(filter_side=conv1_kernel_size, input_tensor=self.x, out_channels=conv1_output_channel, layer_name='conv1',keep_prob = 0.9)
            print("conv1", conv1.shape)
            # convolution, relu:  conv2(?, 16, 16, 16)
            # max pool: conv2(?, 8, 8, 16)
            conv2 = conv_layer(filter_side=conv2_kernel_size, input_tensor=conv1, out_channels=conv2_output_channel, layer_name='conv2', keep_prob = 0.9)
            print("conv2", conv2.shape)
            # conv3 (?, 8, 8, 16)
            # conv3 (?, 4, 4, 16)

            conv3 = conv_layer(filter_side=conv3_kernel_size, input_tensor=conv2, out_channels=conv3_output_channel, layer_name='conv3', keep_prob = 0.9)
            print("conv3", conv3.shape)

            # convolution, relu:  conv3(?, 8, 8, 16)
            # max pool: conv3(?, 4, 4, 16)
            #
            # Flatten
            # Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
            # fc0 (?, 256)
            fc0 = flatten(conv3)

            # fc1(?, 120)
            # fc2(?, 84)
            fc1 = fc_layer(fc0, fc1_out, 'fc1')
            # fc2 = fc_layer(fc1, fc2_out, 'fc2')

            # final fc layer
            self.logits = fc_layer(fc1, n_classes, 'fc2', logitsLayer = True)
            # self.logits = fc_layer(fc2, n_classes, 'fc3', logitsLayer = True)
            with tf.name_scope('loss'):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels,logits=self.logits)
                self.loss = tf.reduce_mean(cross_entropy)
            tf.summary.scalar('loss', self.loss)
            #with tf.name_scope('train'):
            #    train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
            # calculate accuracy
            with tf.name_scope('accuracy'):
                correct_prediction = tf.equal(tf.argmax(self.logits, 1), tf.argmax(one_hot_labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', self.accuracy)
            # merge the summaries
            self.merged = tf.summary.merge_all()
            #print("self.merged", self.merged)









