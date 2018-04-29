import tensorflow as tf
import os
import glob

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from LeNet import LeNet
from sklearn.utils import shuffle
from tfUtil import *
from dataUtil import *
# LeNet here stands for a single layer network , not the actual lenet

def run_training(X_train, y_train, X_valid, y_valid, num_epoch, batch_size, learning_rate, model_save_dir, restorePath=None):
    log_dir = '../result'

    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # build LeNet
    lenet = LeNet(
                 img_w = 32,
                 img_h = 32,
                 img_channel = 1,
                 conv1_kernel_size = 5,
                 conv1_output_channel = 16,

                 conv2_kernel_size = 5,
                 conv2_output_channel = 32,

                 conv3_kernel_size=5,
                 conv3_output_channel=64,

                 fc1_out = 512,
                 n_classes = 43)

    with tf.name_scope("train"):
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(lenet.loss)

    num_examples = X_train.shape[0]

    print("starts training")

    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        # initialize variables
        sess.run(tf.global_variables_initializer())

        # Create a saver.
        # The tf.train.Saver must be created after the variables that you want to restore (or save).
        # Additionally it must be created in the same graph as those variables.


        # restore training
        if restorePath:
            print("restore training")
            var_list = tf.global_variables()
            saver = tf.train.Saver(var_list=var_list)
            saver.restore(sess, tf.train.latest_checkpoint(model_save_dir))
        # start a new saver
        else:
            saver = tf.train.Saver(max_to_keep=1)

        # each epoch will shuffle the entire training data
        for ep in range(num_epoch):
            print("epoch: ", ep)
            X_train, y_train =  shuffle(X_train, y_train)

            # train on each batch
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]

                feed = {lenet.x: batch_x, lenet.labels: batch_y}
                _, loss, summary = sess.run([train_step, lenet.loss, lenet.merged], feed_dict=feed)
                # print("summary", summary)
                # print("offset+num_examples*ep", offset+num_examples*ep)
                train_writer.add_summary(summary, offset+num_examples*ep)

            # test on training data
            # print("loss=", loss)


            # save model
            if ep % 1 == 0:
                print("epoch: ", ep)
                print("loss=", loss)
                # test on validation set
                feed = {lenet.x: X_valid, lenet.labels: y_valid}
                accuracy = sess.run(lenet.accuracy, feed_dict=feed)

                print("accuracy = ", accuracy)
                # Append the step number to the checkpoint name:
                saver.save(sess, model_save_dir + '/my-model', global_step=ep)

def test(model_save_dir, X_test, y_test):


    # load the graph structure from the ".meta" file into the current graph.
    tf.reset_default_graph()
    lenet = LeNet(img_w=32,
                  img_h=32,
                  img_channel=1,
                  n_classes=43)

    # load the values of variables.
    # values only exist within a session
    # evaluate the model
    with tf.Session() as sess:
        var_list = tf.global_variables()

        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, tf.train.latest_checkpoint('../model/lenet5/'))

        feed = {lenet.x: X_test, lenet.labels: y_test}
        test_accuracy = sess.run(lenet.accuracy, feed_dict=feed)
        print("Test Accuracy = {:.3f}".format(test_accuracy))


def main():
    num_epoch = 102
    batch_size = 128
    lr = 0.001
    model_save_dir = '../model/lenet5/'

    X_train, y_train, X_valid, y_valid, X_test, y_test = prepareDataPipeline()

    run_training(X_train, y_train, X_valid, y_valid, num_epoch, batch_size, lr, model_save_dir)
    #test(X_test, y_test, model_save_dir)

if __name__ == '__main__':
    main()
