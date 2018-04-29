import cv2
import pickle

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from skimage import exposure


class DataSetTools():
    def __init__(self):
        print("Step 1: Dataset Summary & Exploration")


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

class ImgPreprocess():
    def __init__(self):
        self.a = 1
    def imgStats(self):
        pass
    def preprocess(self):
        pass

