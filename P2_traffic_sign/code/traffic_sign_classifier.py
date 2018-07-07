from LeNet import LeNet

from utils import DataSetTools, TrainMonitorTools
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import glob
from keras.models import load_model
import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint


NumEpochs = 20 # train for 20 epochs
Patience = 5 # early stop training if evaluation performance is not increasing after 5 epochs

class Pipeline():
    def __init__(self, data_dir, visualize_dir, train_model_path='.', test_model_path='.'):
        # img_params = img.shape()
        self.data_dir = data_dir
        self.visualize_dir = visualize_dir
        self.train_model_path = train_model_path
        self.test_model_path = test_model_path
        self.batch_size = 128

    def load_explore_dataset(self):
        # load data
        self.dataTool = DataSetTools(self.data_dir)

        self.num_classes, self.img_width, self.img_height, img_channels = self.dataTool.load_explore()
        self.img_params = [self.img_width, self.img_height, img_channels]

    def buildNetwork(self):
        self.lenet = LeNet(self.num_classes, self.img_params)
        self.lenetModel = self.lenet.build()


    def train(self, numEpochs):
        self.trainMonitTool = TrainMonitorTools()
        one_hot_y_train = np_utils.to_categorical(self.dataTool.y_train, self.num_classes)  # One-hot encode the labels
        one_hot_y_valid = np_utils.to_categorical(self.dataTool.y_valid, self.num_classes)  # One-hot encode the labels
        
        train_generator = self.dataTool.data_generator.flow(self.dataTool.X_train, one_hot_y_train,
                                                            batch_size=self.batch_size)
        valid_generator = self.dataTool.data_generator.flow(self.dataTool.X_valid, one_hot_y_valid,
                                                            batch_size=self.batch_size)
        # monitor the validation accuracy at each epoch
        # and after the validation accuracy has not improved after two epochs, training is interrupted
        # only best model is saved (without save_best_only, model 2 epochs after the best will be saved)
        callbacks = [EarlyStopping(monitor='val_acc', patience=Patience),
                     ModelCheckpoint(filepath=self.train_model_path, monitor='val_acc', save_best_only=True)]
        history = self.lenetModel.fit_generator(train_generator,
                                                epochs=numEpochs,
                                                callbacks=callbacks,
                                                validation_data = valid_generator)
        # monit training and validation loss
        print("Visualize training loss and accuracy change")
        self.trainMonitTool.visualizeTrain(self.visualize_dir, history)

    def test(self, test_data_dir, test_labels):
        files = sorted(glob.glob(test_data_dir+'*.jpg'))
        test_imgs = []
        for f in files:
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.img_width, self.img_height)) # (32, 32, 3)
            test_imgs.append(img)

        test_imgs = np.array(test_imgs)
        test_generator = self.dataTool.data_generator.flow(test_imgs, shuffle=False)
        print("visualization for testing")
        print("visualize before data augmentation on testing data")
        self.dataTool.visualizeUniqueImgs(test_labels, test_imgs, tag='test',
                                          numImgs = len(test_imgs), numRows = 1)
        print("visualize after data augmentation on testing data")
        self.dataTool.visualizeUniqueImgs(test_labels, test_imgs, tag='test_aug',
                                          numImgs = len(test_imgs), numRows = 1,
                                          data_generator=self.dataTool.data_generator)
        test_model = load_model(self.test_model_path)
        pred_prob = test_model.predict_generator(test_generator) # (5, 43)

        # generate top 5 probabilities
        self.dataTool.top_k(pred_prob, test_labels, k=5)


def main():
    data_dir = "../data/"
    visualize_dir =  data_dir+'visualize/'
    test_data_dir = data_dir + 'testImg/'
    model_path = data_dir+'model/trafficSign_model.h5'
    test_labels = [13, 3, 14, 27, 40]
    traffic_sign_pipeline = Pipeline(data_dir, visualize_dir, train_model_path= model_path, test_model_path= model_path)
    print("Step 1: Dataset Summary & Exploration")
    traffic_sign_pipeline.load_explore_dataset()
    # print("Step 2: Build network")
    # traffic_sign_pipeline.buildNetwork()
    # print("Step 3: Train network")
    # traffic_sign_pipeline.train(numEpochs=NumEpochs)
    print("Step 4: Test on new testing images from outside the dataset")
    traffic_sign_pipeline.test(test_data_dir, test_labels)

if __name__ == '__main__':
    main()

