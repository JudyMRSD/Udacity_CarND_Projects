import cv2
import pickle
import collections

import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
plt.switch_backend('agg')
from skimage import exposure
from collections import defaultdict

class TrainMonitorTools():
    def __init__(self):
        print("train process tool")
    def visualizeTrain(self, visualize_dir, history):
        # plot code from tutorial: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
        print(history.history.keys())  # dict_keys(['val_acc', 'acc', 'loss', 'val_loss'])
        plt.close('all')
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(visualize_dir+'acc.jpg')
        # plt.show()
        # summarize history for loss
        plt.close('all')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        plt.savefig(visualize_dir+'loss.jpg')

class DataSetTools():
    def __init__(self, data_dir):
        print("Step 1: Dataset Summary & Exploration")
        self.X_train_augment = None
        self.data_dir = data_dir
        self.ground_truth_dir = self.data_dir + 'dataset_groundtruth/'
        self.visualize_dir=self.data_dir+'visualize/'


    def loadData(self):
        print("load trianing and validation data")
        training_file = self.ground_truth_dir + "train.p"
        validation_file = self.ground_truth_dir + "valid.p"
        testing_file = self.ground_truth_dir + "test.p"

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        self.X_train, self.y_train = train['features'], train['labels']
        self.X_valid, self.y_valid = valid['features'], valid['labels']
        self.X_test, self.y_test = test['features'], test['labels']

        self.X_train_gray = self.gray(self.X_train)
        self.X_valid_gray = self.gray(self.X_valid)


    def summarizeData(self):
        self.n_train = self.X_train.shape[0]
        self.n_test = self.X_test.shape[0]
        self.image_shape = self.X_train[0].shape

        self.n_classes = np.unique(self.y_train).shape[0]

        print("Number of training examples =", self.n_train)
        print("Number of testing examples =", self.n_test)
        print("Image data shape =", self.image_shape)
        print("Number of classes =", self.n_classes)

    def visualizeHistogram(self, labels, fileName):
        # histogram of classes
        plt.close('all')
        print("historgram bins arranged by classes ")
        plt.hist(labels, bins=self.n_classes)
        plt.title(fileName + "Data Histogram")
        plt.xlabel("Class")
        plt.ylabel("Occurence")
        plt.savefig(self.visualize_dir + fileName + '_histogram.jpg')
        # plt.show()

    def visualizeUniqueImgs(self, labels, imgs, tag, isGray):
        # plot unique images

        numRows = 1

        if tag == 'train':
            print("isGray, train", isGray)
            _, unique_indices = np.unique(labels, return_index=True)
            unique_images = imgs[unique_indices]
            numImgs = self.n_classes
            numRows = 5
        elif tag == 'augment':
            print("isGray, augment", isGray)
            img = np.expand_dims(imgs[0], 0)
            label = labels[0:1]
            unique_images = []
            numImgs = 10
            numRows = 5

            for i in range(0,numImgs):
                aug_img, aug_label = self.train_datagen.flow(img, label).next()
                aug_img = np.squeeze(aug_img)
                unique_images.append(aug_img)
                # print("unique_images", len(unique_images))

        elif tag == 'test':
            print("test visualize")
            numRows = 5
            unique_images = imgs
            numImgs = len(unique_images)
            print("numImgs", numImgs)

        fig = plt.figure()
        for i in range(numImgs):
            ax = fig.add_subplot(numRows, self.n_classes / numRows + 1, i + 1, xticks=[], yticks=[])
            ax.set_title(i)
            if isGray == True:
                ax.imshow(unique_images[i], cmap='gray')
            else:
                ax.imshow(unique_images[i])

        plt.savefig(self.visualize_dir + tag + '_sample')
        # plt.show()
        # plt.close('all')


    def visualizeData(self, tag):
        # two options, training visualize and augmented data visualize
        if tag == 'train':
            isGray = False
            imgs, labels = self.X_train, self.y_train
        elif tag == 'augment':
            isGray = True
            imgs, labels = self.X_train_augment, self.y_train

        self.visualizeHistogram(labels, tag)
        self.visualizeUniqueImgs(labels, imgs, tag, isGray)

    def data_augment(self):
        # balance using keras ImageDataGenerator
        self.train_datagen =ImageDataGenerator(
                        data_format='channels_last',
                        rotation_range=15,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.2,
                        horizontal_flip=True)
        self.visualizeUniqueImgs(self.y_train, self.X_train_gray, tag='augment', isGray=True)

    def gray(self, X):
        # shape is tuple, not mutable
        print("x shape", X.shape) # (34799, 32, 32, 3)
        singleChannelShape = X.shape[0:3]
        # set to single channel
        X_singleChannel = np.zeros(singleChannelShape)
        print("X_singleChannel shape", X_singleChannel.shape) # (34799, 32, 32)

        for i in range(0, len(X)):
            img = X[i]
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            X_singleChannel[i] = gray_img

        X_singleChannel =  np.expand_dims(X_singleChannel, axis=3) # (34799, 32, 32, 1)
        print("X_singleChannel shape", X_singleChannel.shape)
        return X_singleChannel
