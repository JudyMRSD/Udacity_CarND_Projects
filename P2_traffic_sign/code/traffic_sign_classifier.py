# todo: add early stopping uisng validation accuracy
# https://stackoverflow.com/questions/43906048/keras-early-stopping


from LeNet import LeNet
from utils import DataSetTools
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from keras.callbacks import History

class Pipeline():
    def __init__(self, data_dir, visualize_dir):
        # img_params = img.shape()
        self.data_dir = data_dir
        self.visualize_dir = visualize_dir

        self.batch_size = 128

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
        print("img_width, img_height, img_channels", img_width, img_height, img_channels)
        # gray:
        img_channels = 1
        self.img_params = [img_width, img_height, img_channels]

    def buildNetwork(self):
        self.lenet = LeNet(self.num_classes, self.img_params)
        self.lenetModel = self.lenet.build()


    def train(self):
        one_hot_y_train = np_utils.to_categorical(self.dataTool.y_train, self.num_classes)  # One-hot encode the labels
        # train using gray images
        train_generator = self.dataTool.train_datagen.flow(self.dataTool.X_train_gray, one_hot_y_train, batch_size=self.batch_size)
        print("train_generator", train_generator)
        #history = History()
        self.lenetModel.fit_generator(train_generator,
                                           epochs=20)
        #                                   callbacks = [history])
        # loss = history.history['loss']
        # print("loss", loss)

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

