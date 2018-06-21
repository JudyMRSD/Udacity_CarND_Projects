#
# visualize data: distribution of steering angles
import cv2
import os

import numpy as np
import sklearn
import csv
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers import Lambda, Flatten, Dense
import cv2
import numpy as np

class ModelUtil():
    # input : RGB image, output: steering angle
    def __init__(self):
        self.a = 0

    @staticmethod
    def create_network(top_crop, bottom_crop, input_shape):
        # set up cropping2D layer
        model = Sequential()
        model.add(Cropping2D(cropping=((top_crop, bottom_crop),(0,0)), input_shape=input_shape))
        # From Udacity online course: add lambda layer to normalize image and bring to zero mean
        model.add(Lambda(lambda x:(x/255.0)-0.5))
        model.add(Flatten())
        model.add(Dense(1)) # output steering angle
        model.summary()
        model.compile(loss='mse', optimizer='adam')
        return model

class DataUtil():
    def __init__(self):
        pass

    def generator(self, image_dir, samples, batch_size=32):
        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                angles = []
                for batch_sample in batch_samples:
                    name = image_dir + batch_sample[0].split('/')[-1]
                    print("name",name)
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    print("center_image", center_image.shape)
                    angles.append(center_angle)

                # trim image to only see section with road

                X_train = np.array(images)
                Y_train = np.array(angles)
                print("X_train.shape", X_train.shape)
                print("Y_train.shape", Y_train.shape)
                yield sklearn.utils.shuffle(X_train, Y_train)

    def train_val_generator(self, image_dir, csv_path):
        samples = []
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)
        train_samples, validation_samples = train_test_split(samples, test_size=0.2)
        num_train_samples = len(train_samples)
        num_validation_samples = len(validation_samples)
        train_generator = self.generator(image_dir, train_samples, batch_size=32)

        validation_generator = self.generator(image_dir, validation_samples, batch_size=32)

        return  num_train_samples, num_validation_samples, train_generator, validation_generator

    """
    If the above code throw exceptions, try
    model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
    validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
    """