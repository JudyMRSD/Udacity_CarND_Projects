# todo: plot loss curve
# input RGB image, output steer angle

# crop layer example from Udacity
# crop 50 rows piexels from top of the image and 20 rows from bottom of the image
Top_Crop = 60
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
Num_Epochs = 17
Vis = True # visualize output for debugging
Batch_size = 25  # 50 in the example
Num_Val_Samples = 1000 # param from example
Num_Train_Samples = 32000  # param from example
Model_Name = ModelDir+'july8_model.h5'
class Pipeline():
    def __init__(self):
        print("init")
        self.learning_rate = 0.001
        self.modelUtil = ModelUtil()
        self.dataUtil = DataUtil()
        self.visUtil = VisualizeUtil()
        print(self.learning_rate)

    def train(self):
        num_train_samples, num_validation_samples, train_generator, validation_generator = \
            self.dataUtil.train_val_generator(csv_path = Driving_Log_Path, image_dir=Img_Data_Dir,
                                              debug_dir = Debug_Dir, batch_size=Batch_size)

        # if Vis:
        #     self.visUtil.vis_generator(train_generator, "remove_small_angles", save_dir=Debug_Dir)

        print("build model")
        model = self.modelUtil.create_network(Top_Crop, Bottom_Crop, Input_Shape)
        # TODO: add callbacks
        # TODO: add data augmentation
        # patience = 20 essentially turned off patience param
        print("save to : ", Model_Name)
        callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                     ModelCheckpoint(filepath=Model_Name, monitor='val_loss', save_best_only=True)]
        print("num_train_samples",num_train_samples)
        print("num_validation_samples", num_validation_samples)

        model.fit_generator(train_generator,
                            samples_per_epoch= Num_Train_Samples,
                            validation_data=validation_generator,
                            nb_val_samples=Num_Val_Samples,
                            nb_epoch=Num_Epochs,
                            callbacks=callbacks,
                            verbose=1)
        model.save(Model_Name)
        # TODO: test on X_test, y_test, print accuracy

def main():
    print("main function from model.py")
    pl = Pipeline()
    pl.train()



if __name__ == '__main__':
    main()