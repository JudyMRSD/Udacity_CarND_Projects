import cv2
import pickle

import numpy as np
import matplotlib.pyplot as plt
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

    def visualizeUniqueImgs(self, labels, imgs, isGray, imgPath, fileName):
        # plot unique images
        numRows = 5
        _, unique_indices = np.unique(labels, return_index=True)
        unique_images = imgs[unique_indices]
        fig = plt.figure()
        for i in range(self.n_classes):
            ax = fig.add_subplot(numRows, self.n_classes / numRows + 1, i + 1, xticks=[], yticks=[])
            ax.set_title(i)
            if (isGray == True):
                ax.imshow(np.squeeze(unique_images[i]), cmap='gray')
            else:
                ax.imshow(unique_images[i])

        plt.savefig(imgPath + fileName + '_sample')
        plt.close('all')


    def visualizeData(self, tag, imgPath):
        # two options, training visualize and augmented data visualize
        if tag == 'train':
            isGray = False
            imgs, labels, fileName = self.X_train, self.y_train, 'training'
        elif tag == 'augment':
            isGray = True
            imgs, labels, fileName = self.X_train_augment, self.y_train, 'augmented'

        self.visualizeHistogram(labels, fileName, imgPath)
        self.visualizeUniqueImgs(labels, imgs, isGray, imgPath, fileName)

class ImgPreprocess():
    def __init__(self):
        self.a = 1
    def imgStats(self):
        pass
    def preprocess(self):
        pass

