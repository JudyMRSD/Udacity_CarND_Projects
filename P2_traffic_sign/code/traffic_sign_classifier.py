from LeNet import LeNet
from utils import DataSetTools
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values

class Pipeline():
    def __init__(self, data_dir, visualize_dir):
        # img_params = img.shape()
        self.data_dir = data_dir
        self.visualize_dir = visualize_dir
        self.exploreDataset()
        # todo: uncomment buildModel:
        # self.buildNetwork()
        self.batch_size = 1

    def exploreDataset(self):
        # load data
        self.dataTool = DataSetTools()
        self.dataTool.loadData(self.data_dir)
        self.dataTool.summarizeData()
        self.dataTool.visualizeData(tag='train', imgPath=self.visualize_dir)
        self.dataTool.data_augment()
        # summarize dataset info
        self.num_classes = self.dataTool.n_classes
        img_width, img_height, img_channels = self.dataTool.image_shape
        self.img_params = [img_width, img_height, img_channels]

    def buildNetwork(self):
        self.lenet = LeNet(self.num_classes, self.img_params)
        self.lenetModel = self.lenet.build()


    def train(self):
        one_hot_y_train = np_utils.to_categorical(self.dataTool.y_train, self.num_classes)  # One-hot encode the labels

        train_generator = self.dataTool.train_datagen.flow(self.dataTool.X_train, one_hot_y_train, batch_size=32)
        print("train_generator", train_generator)
        history = self.lenetModel.fit_generator(train_generator,
                                           steps_per_epoch= 32,
                                           epochs=20)

    def classify(self):
        pass



def main():
    # set input training image directory
    # run pipeline
    # output classification results
    data_dir = '../traffic-signs-data/'
    visualize_dir =  '../visualize/'
    traffic_sign_pipeline = Pipeline(data_dir, visualize_dir)

    traffic_sign_pipeline.exploreDataset()
    traffic_sign_pipeline.buildNetwork()
    traffic_sign_pipeline.train()
    # traffic_sign_pipeline.classify()


if __name__ == '__main__':
    main()

