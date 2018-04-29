from LeNet import LeNet
from utils import DataSetTools

class Pipeline():
    def __init__(self, data_dir, visualize_dir):
        # img_params = img.shape()
        self.data_dir = data_dir
        self.visualize_dir = visualize_dir
        self.exploreDataset()
        # todo: uncomment buildModel:
        # self.buildNetwork()

    def exploreDataset(self):
        # load data
        self.dataTool = DataSetTools()
        self.dataTool.loadData(self.data_dir)
        self.dataTool.summarizeData()
        self.dataTool.visualizeData(tag='train', imgPath=self.visualize_dir)
        self.dataTool.balanceClass()
        self.dataTool.visualizeDatagen()
        # summarize dataset info
        self.num_classes = self.dataTool.n_classes
        img_width, img_height, img_channels = self.dataTool.image_shape
        self.img_params = [img_width, img_height, img_channels]

    def buildNetwork(self):
        self.lenet = LeNet(self.num_classes, self.img_params)
        self.lenet.build()


    def train(self):
        pass

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

    # traffic_sign_pipeline.train()
    # traffic_sign_pipeline.classify()


if __name__ == '__main__':
    main()

