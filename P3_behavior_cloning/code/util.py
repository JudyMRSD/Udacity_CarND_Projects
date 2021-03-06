import sklearn
import csv
from sklearn.model_selection import train_test_split

from keras.layers import Cropping2D
import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from keras.layers import Conv2D, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Dense
from keras.models import Sequential

NumSamples = -1  # -1  use all samples, 32 to debug
Vis = True # visualize output for debugging
Angle_Thresh = 0.01 # angles less than Angle_Thresh will be kept with Keep_Zero_Prob
Keep_Zero_Prob = 0.25

class ModelUtil():
    # input : RGB image, output: steering angle
    def __init__(self):
        self.a = 0

    def build_conv_layers(self, num_filter_list, kernel_size_list, activation ='elu', pool_size=2, dropout_ratio=0.3):
        for i in range(len(num_filter_list)):
            self.model.add(Conv2D(num_filter_list[i], kernel_size_list[i], kernel_size_list[i], activation=activation))
            self.model.add(MaxPooling2D(pool_size=(pool_size,pool_size)))  # default 2x2 pooling
            self.model.add(Dropout(dropout_ratio))

    # make regular densely-connected NN-layer ,  output = activation(dot(input, kernel) + bias)
    def build_dense_layers(self, units_list, activation = 'elu', dropout_ratio=0.5):
        for i in range(len(units_list)):
            self.model.add(Dense(units_list[i], activation = activation))
            self.model.add(Dropout(dropout_ratio))


    def create_network(self, top_crop, bottom_crop, input_shape):

        # started with the architecture here, but I implemented the code myself
        # https://github.com/mvpcom/Udacity-CarND-Project-3/blob/master/model.ipynb
        self.model = Sequential()
        # preprocess layers to get region of interest ((top_crop, bottom_crop), (left_crop, right_crop))
        self.model.add(Cropping2D(cropping=((top_crop, bottom_crop), (0, 0)), input_shape=input_shape))
        # normalize input
        self.model.add(Lambda(lambda x: (x / 255.0) - 0.5))
        # conv layers
        self.model.add(Conv2D(3, 1, 1,activation ='elu', name='conv_1'))
        self.build_conv_layers(num_filter_list=[16, 32, 32], kernel_size_list=[8, 5, 3])

        self.model.add(Flatten())

        self.build_dense_layers(units_list=[256, 128, 64, 8])
        self.model.add(Dense(1, activation = 'elu', name='output_angle'))

        self.model.compile(loss='mse', optimizer='adam')
        self.model.summary()
        return self.model

class DataUtil():
    def __init__(self):
        self.center_col_idx = 0
        self.left_col_idx = 1
        self.right_col_idx = 2
        self.center_angle_col_idx = 3
        self.angle_correction = 0.2
        self.visUtil = VisualizeUtil()
    # for each set of center, left and right images, adjust center angle for left and right image
    # each of the 3 images are flipped to balance left and right turns
    # shift is applied to simulate cars at different locations in the lane
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
                # resize to half width and height
                # idea from:  https://medium.com/@ValipourMojtaba/my-approach-for-project-3-2545578a9319
                bgr_image = cv2.resize(bgr_image, (0, 0), fx=0.5, fy=0.5)
                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)  # convert color to RGB to match drive.py
                adj_ang = angle+angle_adjust[i]
                # data augmentation
                flipped_img, flipped_ang = self.aug_flip(rgb_image, adj_ang)

                shifted_imgs, shifted_angs = self.aug_shift(flipped_img, flipped_ang)
                images.extend([rgb_image, flipped_img])
                angles.extend([adj_ang, flipped_ang])

                images.extend(shifted_imgs)
                angles.extend(shifted_angs)
        # no data agumentation on validation set
        else:
            center_img_name = self.image_dir + batch_sample[self.center_col_idx].split('/')[-1]
            bgr_image = cv2.imread(center_img_name)

            bgr_image = cv2.resize(bgr_image, (0,0), fx=0.5, fy=0.5)

            angle = float(batch_sample[self.center_angle_col_idx])
            images.append(bgr_image)
            angles.append(angle)

        return images, angles

    def aug_flip(self, img, angle):
        # (left turn -> right turn)
        img = cv2.flip(img, 1)
        angle = -angle
        return img, angle


    def aug_shift(self, image, angle):
        # idea from https://medium.com/@ValipourMojtaba/my-approach-for-project-3-2545578a9319
        shifted_images = []
        shifted_angles = []
        for i in range (self.num_shift):
            # my code
            img_h, img_w, _ = image.shape
            x_shift_range = 100 # 200
            trans_x = x_shift_range * (np.random.rand() - 0.5)
            y_shift_range = 10 # 20
            trans_y = y_shift_range * (np.random.rand() - 0.5)
            # for every pixel shift in x direction, change angle by angle_per_pix
            angle_per_pix = 0.8 # 0.4
            shifted_ang = angle + trans_x/x_shift_range * angle_per_pix
            affine_matrix = np.array([[1, 0 , trans_x],
                                      [0, 1, trans_y]], dtype = np.float32)
            shifted_img = cv2.warpAffine(image, affine_matrix, (img_w, img_h))
            if abs(shifted_ang) <= 1: #  steering angle range -1 : 1
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

    def less_zero(self, angles):
        angles = np.array(angles)
        zero_thresh = 0.01
        zero_angle_index = angles < zero_thresh
        zero_angle_index[25:] = False # since samples are already shuffled, only take first 25% of zero_angles
        non_zero_angle_index = angles > zero_thresh
        selected_index = zero_angle_index | non_zero_angle_index # take valid zero angle and non zero angle
        return selected_index
                
    def generator(self, samples, is_train, batch_size=10):
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

    def train_val_generator(self, csv_path,image_dir, debug_dir, batch_size):
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

        # only keep 0.25 of zero angles
        np_sample = np.array(samples)
        angle_column = np.array(np_sample[:, self.center_angle_col_idx], dtype= float)
        non_zero_angles_index = np.absolute(angle_column) > Angle_Thresh
        zero_angles = np.absolute(angle_column) < Angle_Thresh
        zeros_prob_angle = np.random.random_sample((len(zero_angles),)) < Keep_Zero_Prob
        zero_angles_index = np.logical_and(zeros_prob_angle , zero_angles)

        valid_sample_idx = np.logical_or(zero_angles_index, non_zero_angles_index)
        filtered_samples = np_sample[valid_sample_idx, :]

        if Vis:
            aug_imgs, aug_angles = self.sample_img_ang(filtered_samples[10], is_train=True)
            self.visUtil.vis_img_aug(aug_imgs, aug_angles, self.debug_dir)

        train_samples, validation_samples = train_test_split(samples, test_size=0.2)
        num_train_samples = len(train_samples)
        num_validation_samples = len(validation_samples)

        print("create generator")
        print("batch_size", batch_size)
        train_generator = self.generator(train_samples, is_train = True, batch_size=batch_size)

        validation_generator = self.generator(validation_samples,  is_train=False, batch_size=batch_size)

        return  num_train_samples, num_validation_samples, train_generator, validation_generator


class VisualizeUtil():
    def __init__(self):
        pass

    def vis_generator(self, generator, name, save_dir):
        images, angles = generator.__next__()
        plt.hist(angles, bins='auto')
        plt.title(name)
        plt.xlabel("angle")
        plt.ylabel("count")
        plt.savefig(save_dir + name + ".png")
        # 4: original, flipped, shifted
        ax_row = 3
        ax_col = 1
        fig, ax = plt.subplots(ax_row, ax_col, figsize=(16, 9))
        img_vis = images[:ax_row]
        for i, img in enumerate(img_vis):
            ax[i].imshow(img)
        plt.savefig(save_dir + "vis_aug_imgs.jpg")

    def draw_line(self, row, col, name, img, angle):
        # steering angle is between -1 and 1
        # convert -1 to 1  to angles  (angle + 1)/2 ---- [0,1]
        # turn left: +    turn right: -
        self.ax[row][col].set_title(name+",  angle "+ "{0:.2f}".format(round(angle,2)))
        h, w, _ = img.shape
        # keep the line pointing from center of bottom of frame to the middle of image, inspired by jeremy-shannon
        cv2.line(img, (int(w / 2), int(h)), (int(w / 2 + angle * w / 4), int(h / 2)), (0, 255, 0),
                 thickness=4)

        self.ax[row][col].imshow(img)

    def vis_img_aug(self, aug_imgs, aug_angles, save_dir):
        print("aug_imgs", len(aug_imgs))
        images = aug_imgs[0:4]

        ax_row = 2
        ax_col = 2
        fig, self.ax = plt.subplots(ax_row, ax_col, figsize=(16, 9))
        self.draw_line(0, 0, "original", images[0], aug_angles[0])
        self.draw_line(0, 1, "flipped", images[1], aug_angles[1])
        self.draw_line(1, 0, "adjust_light", images[2], aug_angles[2])
        self.draw_line(1, 1, "shifted", images[3], aug_angles[3])

        plt.savefig(save_dir + "vis_aug_imgs.jpg")
        plt.close()
        print("finished save image")






























