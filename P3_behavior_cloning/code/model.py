
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
from util import ModelUtil, DataUtil, VisualizeUtil
from keras.callbacks import EarlyStopping, ModelCheckpoint

# TODO: load image as RGB (drive.py use RGB)
ModelDir = "../data/model/"
Driving_Log_Path = "../data/driving_log.csv"
Img_Data_Dir = "../data/IMG/"
Debug_Dir = "../data/debug/"
Num_Epochs = 5
Vis = True # visualize output for debugging

class Pipeline():
    def __init__(self):
        print("init")
        self.learning_rate = 0.001
        self.modelUtil = ModelUtil()
        self.dataUtil = DataUtil()
        self.visUtil = VisualizeUtil()
        print(self.learning_rate)

    def train(self, train_model_path):
        num_train_samples, num_validation_samples, train_generator, validation_generator = \
            self.dataUtil.train_val_generator(csv_path = Driving_Log_Path, image_dir=Img_Data_Dir, debug_dir = Debug_Dir)

        if Vis:
            self.visUtil.vis_generator(train_generator, "remove_small_angles", save_dir=Debug_Dir)

        print("build model")
        model = self.modelUtil.create_network(Top_Crop, Bottom_Crop, Input_Shape)
        # TODO: add callbacks
        # TODO: add data augmentation
        callbacks = [EarlyStopping(monitor='val_loss', patience=2),
                     ModelCheckpoint(filepath=train_model_path, monitor='val_loss', save_best_only=True)]
        print("num_train_samples",num_train_samples)
        print("num_validation_samples", num_validation_samples)

        model.fit_generator(train_generator,
                            samples_per_epoch= num_train_samples,
                            validation_data=validation_generator,
                            nb_val_samples=num_validation_samples,
                            nb_epoch=Num_Epochs,
                            callbacks=callbacks,
                            verbose=2)
        model.save(ModelDir + 'model.h5')
        # TODO: test on X_test, y_test, print accuracy

def main():
    print("main function from model.py")
    pl = Pipeline()
    pl.train(train_model_path= ModelDir + "model.h5")



if __name__ == '__main__':
    main()