# input RGB image, output steer angle

# crop layer example from Udacity
# crop 50 rows piexels from top of the image and 20 rows from bottom of the image
Top_Crop = 50
Bottom_Crop = 20
# Input_Shape = (3,160,320)
Input_Shape = (160,320,3)

from keras.models import Sequential, Model
from keras.layers import Cropping2D
from keras.layers import Lambda, Flatten, Dense
import cv2
import numpy as np
from util import DataUtil
# TODO: load image as RGB (drive.py use RGB)




ModelDir = "../data/model/"
Driving_Log_Path = "../data/driving_log.csv"
Img_Data_Dir = "../data/IMG/"
class DriveModel():
    # input : RGB image, output: steering angle
    def __init__(self):
        self.a = 0

    @staticmethod
    def create_network():
        # set up cropping2D layer
        model = Sequential()
        model.add(Cropping2D(cropping=((50,20),(0,0)), input_shape=Input_Shape))
        # From Udacity online course: add lambda layer to normalize image and bring to zero mean
        model.add(Lambda(lambda x:(x/255.0)-0.5))
        model.add(Flatten())
        model.add(Dense(1)) # output steering angle
        model.summary()
        model.compile(loss='mse', optimizer='adam')
        return model

class pipeline():
    def __init__(self):
        self.learning_rate = 0.001
        self.model = DriveModel()
        self.dataUtil = DataUtil()
        self.X_train, self.Y_train, self.X_test, self.Y_test = \
            self.dataUtil.create_dataset(Img_Data_Dir, Driving_Log_Path)


    def train(self):
        X_train, X_val, Y_train, Y_val, X_test, Y_test = self.dataUtil.split()
        self.model.fit(X_train, Y_train, validation_split = 0.2, shuffle=True, nb_epoch=2)
        self.model.save(ModelDir + 'model.h5')





def main():
    print("main")
    drive_m = drive_model()
    drive_m.create_network()

