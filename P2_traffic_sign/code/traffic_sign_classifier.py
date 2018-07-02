# todo: add early stopping uisng validation accuracy
# https://stackoverflow.com/questions/43906048/keras-early-stopping


from LeNet import LeNet
import matplotlib.pyplot as plt

from utils import DataSetTools, TrainMonitorTools
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.callbacks import History
import glob
from keras.models import load_model
import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint


class Pipeline():
    def __init__(self, data_dir, visualize_dir, train_model_path='.', test_model_path='.'):
        # img_params = img.shape()
        self.data_dir = data_dir
        self.visualize_dir = visualize_dir
        self.train_model_path = train_model_path
        self.test_model_path = test_model_path
        self.batch_size = 128

    def exploreDataset(self):
        # load data
        self.dataTool = DataSetTools(self.data_dir)
        self.dataTool.loadData()
        self.dataTool.summarizeData()
        self.dataTool.visualizeData(tag='train')
        self.dataTool.data_augment()
        # summarize dataset info
        self.num_classes = self.dataTool.n_classes
        self.img_width, self.img_height, img_channels = self.dataTool.image_shape
        print("img_width, img_height, img_channels", self.img_width, self.img_height, img_channels)
        # gray:
        img_channels = 1
        self.img_params = [self.img_width, self.img_height, img_channels]

    def buildNetwork(self):
        self.lenet = LeNet(self.num_classes, self.img_params)
        self.lenetModel = self.lenet.build()


    def train(self, numEpochs):
        self.trainMonitTool = TrainMonitorTools()
        one_hot_y_train = np_utils.to_categorical(self.dataTool.y_train, self.num_classes)  # One-hot encode the labels
        one_hot_y_valid = np_utils.to_categorical(self.dataTool.y_valid, self.num_classes)  # One-hot encode the labels

        # train using gray images
        train_generator = self.dataTool.train_datagen.flow(self.dataTool.X_train_gray, one_hot_y_train, batch_size=self.batch_size)
        validation_XY = (self.dataTool.X_valid_gray, one_hot_y_valid)
        print("train_generator", train_generator)
        #history = History()

        # monitor the test (validation) loss at each epoch
        # and after the test loss has not improved after two epochs, training is interrupted
        # only best model is saved (without save_best_only, model 2 epochs after the best will be saved)
        


        callbacks = [EarlyStopping(monitor='val_acc', patience=20),
                     ModelCheckpoint(filepath=self.train_model_path, monitor='val_loss', save_best_only=True)]
        # # steps_per_epoch
        
        history = self.lenetModel.fit_generator(train_generator,
                                      epochs=numEpochs,
                                      callbacks= callbacks,
                                      validation_data = validation_XY)

        # no callback
        # history = self.lenetModel.fit_generator(train_generator,
        #                               epochs=numEpochs,
        #                               validation_data = validation_XY)
        # model.save(self.train_model_path)
        self.trainMonitTool.visualizeTrain(self.visualize_dir, history)


    def test(self, test_data_dir, test_labels):
        print("enter test:")
        files = sorted(glob.glob(test_data_dir+'*.jpg'))
        color_test_imgs = []
        test_images = []

        for f in files:
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_width, self.img_height)) # (32, 32, 3)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # (32, 32)
            gray_img = np.expand_dims(gray_img, 2)  # (32, 32, 1)
            color_test_imgs.append(img)
            test_images.append(gray_img)
        test_images = np.array(test_images)
        print("test_images shape", test_images.shape)
        # convert images to a 4D tensor to feed into keras model
        test_model = load_model(self.test_model_path)
        # out = test_model.predict(test_images)
        out_class = test_model.predict_classes(test_images)
        self.dataTool.visualizeUniqueImgs(test_labels, color_test_imgs, tag="test", isGray=False)
        print("test_labels", test_labels)
        print("out class", out_class) # out class [12 25  0 14 13]

def main():
    # set input training image directory
    # run pipeline
    # output classification results
    data_dir = "../data/"
    visualize_dir =  data_dir+'visualize/'
    test_data_dir = data_dir+'googleImg/'
    ground_truth = data_dir+'dataset_groundtruth/'
    model_path = data_dir+'model/trafficSign_model.h5'
    test_labels = [34, 25, 3, 14, 13]
    traffic_sign_pipeline = Pipeline(data_dir, visualize_dir, train_model_path= model_path, test_model_path= model_path)
    traffic_sign_pipeline.exploreDataset()
    traffic_sign_pipeline.buildNetwork()
    traffic_sign_pipeline.train(numEpochs=100)
    traffic_sign_pipeline.test(test_data_dir, test_labels)

if __name__ == '__main__':
    main()

