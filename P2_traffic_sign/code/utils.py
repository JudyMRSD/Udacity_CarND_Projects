import cv2
import pickle
import collections

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
# plt.switch_backend('agg')
from skimage import exposure
from collections import defaultdict


class DataSetTools():
    def __init__(self):
        print("Step 1: Dataset Summary & Exploration")

        self.X_train_augment = None

    def loadData(self, data_dir):
        print("load trianing and validation data")
        training_file = data_dir + "train.p"
        validation_file = data_dir + "valid.p"
        testing_file = data_dir + "test.p"

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        self.X_train, self.y_train = train['features'], train['labels']
        self.X_valid, self.y_valid = valid['features'], valid['labels']
        self.X_test, self.y_test = test['features'], test['labels']

    def summarizeData(self):
        self.n_train = self.X_train.shape[0]
        self.n_test = self.X_test.shape[0]
        self.image_shape = self.X_train[0].shape

        self.n_classes = np.unique(self.y_train).shape[0]

        print("Number of training examples =", self.n_train)
        print("Number of testing examples =", self.n_test)
        print("Image data shape =", self.image_shape)
        print("Number of classes =", self.n_classes)

    def visualizeHistogram(self, labels, fileName, imgPath):
        # histogram of classes
        plt.close('all')
        print("historgram bins arranged by classes ")
        plt.hist(labels, bins=self.n_classes)
        plt.title(fileName + "Data Histogram")
        plt.xlabel("Class")
        plt.ylabel("Occurence")
        plt.savefig(imgPath + fileName + '_histogram.jpg')
        # plt.show()

    def visualizeUniqueImgs(self, labels, imgs, tag, imgPath):
        # plot unique images

        numRows = 5

        if tag == 'train':
            _, unique_indices = np.unique(labels, return_index=True)
            unique_images = imgs[unique_indices]
            numImgs = self.n_classes
        elif tag == 'augment':
            img = np.expand_dims(imgs[0], 0)
            label = labels[0:1]
            unique_images = []
            numImgs = 10
            for i in range(0,numImgs):
                aug_img, aug_label = self.train_datagen.flow(img, label).next()
                aug_img = np.squeeze(aug_img)
                unique_images.append(aug_img)
                print("unique_images", len(unique_images))
        fig = plt.figure()
        for i in range(numImgs):
            ax = fig.add_subplot(numRows, self.n_classes / numRows + 1, i + 1, xticks=[], yticks=[])
            ax.set_title(i)
            ax.imshow(unique_images[i])

        plt.savefig(imgPath + tag + '_sample')
        plt.close('all')


    def visualizeData(self, tag, imgPath):
        # two options, training visualize and augmented data visualize
        if tag == 'train':
            imgs, labels = self.X_train, self.y_train
        elif tag == 'augment':
            imgs, labels = self.X_train_augment, self.y_train

        self.visualizeHistogram(labels, tag, imgPath)
        self.visualizeUniqueImgs(labels, imgs, tag, imgPath)

    def data_augment(self):
        # balance using keras ImageDataGenerator
        self.train_datagen =ImageDataGenerator(
                        data_format='channels_last',
                        rotation_range=15,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=0.2,
                        horizontal_flip=True)
        self.visualizeUniqueImgs(self.y_train, self.X_train, tag='augment', imgPath='../visualize/')

    def gray(X):
        threeChannelShape = X.shape
        # shape is tuple, not mutable
        singleChannelShape = threeChannelShape[0:3] + (1,)
        # set to single channel
        X_singleChannel = np.zeros(singleChannelShape)

        for i in range(0, len(X)):
            img = X[i]
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # print("gray shape", gray_img.shape) # (32, 32)
            gray_img = np.expand_dims(gray_img, axis=2)
            # TODO: fix this
            gray_img = normalization(gray_img)

            X_singleChannel[i] = gray_img

            # print("gray img shape",X_singleChannel[i].shape) # (32,32,1)

        # plt.imshow(X[0], cmap='gray')
        # plt.show()

        return X_singleChannel


class ImgPreprocess():
    def __init__(self):
        self.a = 1
    def imgStats(self):
        pass
    def preprocess(self):
        pass
