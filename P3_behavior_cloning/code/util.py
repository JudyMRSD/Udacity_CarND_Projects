#
# visualize data: distribution of steering angles
import cv2
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
import numpy as np
import sklearn
import csv
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers import Lambda, Flatten, Dense
import cv2
import numpy as np

NumSamples = -1  # 32 # -1  use all samples

class ModelUtil():
    # input : RGB image, output: steering angle
    def __init__(self):
        self.a = 0

    @staticmethod
    def create_network(top_crop, bottom_crop, input_shape):
        # set up cropping2D layer
        # model = Sequential()
        # model.add(Cropping2D(cropping=((top_crop, bottom_crop),(0,0)), input_shape=input_shape))
        # # From Udacity online course: add lambda layer to normalize image and bring to zero mean
        # model.add(Lambda(lambda x:(x/255.0)-0.5))
        # model.add(Flatten())
        # model.add(Dense(1)) # output steering angle

        model = Sequential()
        # convolution, relu:  conv1 (?, 32, 32, 6)
        # max pool:  conv1 (?, 16, 16, 6)
        model.add(Conv2D(6, 5, activation='relu', padding='same',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # convolution, relu:  conv2(?, 16, 16, 16)
        # max pool: conv2(?, 8, 8, 16)
        model.add(Conv2D(16, 5, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # conv3 (?, 8, 8, 16)
        # conv3 (?, 4, 4, 16)
        model.add(Conv2D(16, 5, activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # Flatten
        # Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
        # fc0 (?, 256)
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation("relu"))
        # final softmax layer output prediction of probabilities for each class
        model.add(Dense(1))

        model.summary()
        model.compile(loss='mse', optimizer='adam')
        return model

class DataUtil():
    def __init__(self, image_dir):
        self.center_col_idx = 0
        self.left_col_idx = 1
        self.right_col_idx = 2
        self.center_angle_col_idx = 3
        self.image_dir = image_dir
        self.angle_correction = 0.2

    def read_img_angle(self, batch_sample):
        images = []
        angles = []
        col_idx = [self.center_col_idx, self.left_col_idx, self.right_col_idx]
        angle_adjust = [0,  self.angle_correction, -self.angle_correction]
        angle = float(batch_sample[self.center_angle_col_idx])

        for i in col_idx:
            bgr_image = cv2.imread(self.image_dir + batch_sample[i].split('/')[-1])
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # convert color to RGB to match drive.py
            images.append(rgb_image)
            angles.append(angle+angle_adjust[i])
        return images, angles

    def generator(self, samples, batch_size=32):
        num_samples = len(samples)
        print("num_samples", num_samples)

        while 1:  # Loop forever so the generator never terminates
            sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                angles = []
                for batch_sample in batch_samples:
                    sample_images, sample_angles = self.read_img_angle(batch_sample)
                    images.extend(sample_images)
                    angles.extend(sample_angles)
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    def train_val_generator(self, csv_path):
        samples = []
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # skip header
            for line in reader:
                samples.append(line)
        samples=samples[0:NumSamples]
        train_samples, validation_samples = train_test_split(samples, test_size=0.2)
        num_train_samples = len(train_samples)
        num_validation_samples = len(validation_samples)
        train_generator = self.generator(train_samples, batch_size=32)

        validation_generator = self.generator(validation_samples, batch_size=32)

        return  num_train_samples, num_validation_samples, train_generator, validation_generator

    """
    If the above code throw exceptions, try
    model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
    validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
    """