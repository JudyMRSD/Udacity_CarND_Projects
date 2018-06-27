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
import matplotlib.pyplot as plt

NumSamples = -1  # 32 # -1  use all samples

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
        # Lenet
        # model = Sequential()
        # # convolution, relu:  conv1 (?, 32, 32, 6)
        # # max pool:  conv1 (?, 16, 16, 6)
        # model.add(Conv2D(6, 5, activation='relu', padding='same',
        #                  input_shape=input_shape))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # # convolution, relu:  conv2(?, 16, 16, 16)
        # # max pool: conv2(?, 8, 8, 16)
        # model.add(Conv2D(16, 5, activation='relu', padding='same'))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # # conv3 (?, 8, 8, 16)
        # # conv3 (?, 4, 4, 16)
        # model.add(Conv2D(16, 5, activation='relu', padding='same'))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        # # Flatten
        # # Flatten the output shape of the final pooling layer such that it's 1D instead of 3D.
        # # fc0 (?, 256)
        # model.add(Flatten())
        # model.add(Dense(256))
        # model.add(Activation("relu"))
        # # final softmax layer output prediction of probabilities for each class
        # model.add(Dense(1))

        model.summary()
        model.compile(loss='mse', optimizer='adam')
        return model

class DataUtil():
    def __init__(self):
        self.center_col_idx = 0
        self.left_col_idx = 1
        self.right_col_idx = 2
        self.center_angle_col_idx = 3
        self.angle_correction = 0.2

    def sample_img_ang(self, batch_sample, is_train):
        # use center, left and right images by making adjustment ot turning angle
        images = []
        angles = []
        if is_train:
            col_idx = [self.center_col_idx, self.left_col_idx, self.right_col_idx]
            angle_adjust = [0,  self.angle_correction, -self.angle_correction]
            angle = float(batch_sample[self.center_angle_col_idx])

            for i in col_idx:
                bgr_image = cv2.imread(self.image_dir + batch_sample[i].split('/')[-1])
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # convert color to RGB to match drive.py
                adj_ang = angle+angle_adjust[i]
                # data augmentation: flipping half of the images
                flipped_img, flipped_ang = self.aug_flip(rgb_image, adj_ang)
                light_img, light_ang = self.aug_light(flipped_img, flipped_ang)
                shifted_imgs, shifted_angs = self.aug_shift(flipped_img, flipped_ang)

                # save img and angle
                # images.extend([rgb_image, flipped_img])
                # angles.extend([adj_ang, flipped_ang])

                # if (abs(adj_ang) > self.angle_correction): # only keep half of the angles > 0.2
                #     images.extend([rgb_image, flipped_img])
                #     angles.extend([adj_ang, flipped_ang])

                images.extend([rgb_image, flipped_img])
                angles.extend([adj_ang, flipped_ang])
                images.append(light_img)
                angles.append(light_ang)
                images.extend(shifted_imgs)
                angles.extend(shifted_angs)
        # no data agumentation on validation set
        else:
            center_img_name = self.image_dir + batch_sample[self.center_col_idx].split('/')[-1]
            bgr_image = cv2.imread(center_img_name)
            angle = float(batch_sample[self.center_angle_col_idx])
            images.append(bgr_image)
            angles.append(angle)

        return images, angles

    def aug_flip(self, img, angle):
        if np.random.rand()>0.5:
            # (left turn -> right turn)
            img = cv2.flip(img, 1)
            print("if img is none, flip does not return value", img.shape)
            angle = -angle
        return img, angle

    def aug_shift(self, image, angle):
        # took from https://medium.com/@ValipourMojtaba/my-approach-for-project-3-2545578a9319
        # including hyper parameters

        shifted_images = []
        shifted_angles = []
        for i in range (self.num_shift):
            # my code
            img_h, img_w, _ = image.shape
            x_shift_range = 100
            # np.random.rand() - 0.5  random number (0, 1) -> (-0.5 , 0.5)
            trans_x = x_shift_range * (np.random.rand() - 0.5)
            y_shift_range = 5
            trans_y = y_shift_range * (np.random.rand() - 0.5)
            # for every pixel shift in x direction, change angle by angle_per_pix
            angle_per_pix = 0.8
            shifted_ang = angle + trans_x/x_shift_range * angle_per_pix
            affine_matrix = np.array([[1, 0 , trans_x],
                                      [0, 1, trans_y]], dtype = np.float32)
            shifted_img = cv2.warpAffine(image, affine_matrix, (img_w, img_h))
            shifted_images.append(shifted_img)
            shifted_angles.append(shifted_ang)
        return shifted_images, shifted_angles

    def aug_light(self, rgb_img, angle):
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        # Reference: https://github.com/mvpcom/Udacity-CarND-Project-3/blob/master/model.ipynb
        rand_v_channel = 0.25 + np.random.rand()
        hsv_img[:,:,2] = hsv_img[:,:,2]  * rand_v_channel
        aug_light_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
        # angle not changed, directly returned
        return aug_light_img, angle

    def aug_shadow(self):
        # https: // github.com / jeremy - shannon / CarND - Behavioral - Cloning - Project
        pass

    def generator(self, samples, is_train, batch_size=32):
        num_samples = len(samples)
        print("num_samples", num_samples)
        while 1:  # Loop forever so the generator never terminates
            sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]
                images = []
                angles = []
                for batch_sample in batch_samples:
                    sample_images, sample_angles = self.sample_img_ang(batch_sample, is_train)
                    images.extend(sample_images)
                    angles.extend(sample_angles)
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

    def train_val_generator(self, csv_path,image_dir, debug_dir):
        self.image_dir = image_dir
        self.debug_dir = debug_dir
        self.num_shift = 10
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
        train_generator = self.generator(train_samples, is_train = True, batch_size=100)

        validation_generator = self.generator(validation_samples,  is_train=False, batch_size=32)

        return  num_train_samples, num_validation_samples, train_generator, validation_generator

    """
    If the above code throw exceptions, try
    model.fit_generator(train_generator, steps_per_epoch= len(train_samples),
    validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
    """

class VisualizeUtil():
    def __init__(self):
        pass
    def vis_generator(self, dataUtil, generator, name, save_dir):
        tmp_num_shift = dataUtil.num_shift
        dataUtil.num_shift = 1
        images, angles = generator.__next__()
        plt.hist(angles, bins='auto')
        plt.title(name)
        plt.xlabel("angle")
        plt.ylabel("count")
        plt.savefig(save_dir + name + ".png")
        # 4: original, flipped, shifted, adjust light
        # set Num_Shift = 1 to use this part (so the first 4 images
        # are from different modifications on image)
        ax_row = 2
        ax_col = 2
        fig, ax = plt.subplots(ax_row, ax_col, figsize=(16, 9))
        # todo: plot a line indicating angle on the image
        original = images[0]
        flipped = images[1]
        shifted = images[2]
        adjust_light = images[3]
        ax[0][0].set_title("original")
        ax[0][0].imshow(original)
        ax[0][1].set_title("flipped")
        ax[0][1].imshow(flipped)
        ax[1][0].set_title("shifted")
        ax[1][0].imshow(shifted)
        ax[1][1].set_title("adjust_light")
        ax[1][1].imshow(adjust_light)
        plt.savefig(save_dir + "vis_aug_imgs.jpg")

        dataUtil.num_shift = tmp_num_shift





























