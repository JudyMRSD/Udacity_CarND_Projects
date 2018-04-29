from LeNet import LeNet


class Pipeline():
    def __init__(self):
        self.num_classes = 40
        img_width, img_height, img_channels = [32, 32, 1]
        self.img_params = [img_width, img_height, img_channels]
        # img_params = img.shape()
        self.lenet = LeNet(self.num_classes, self.img_params)
        print("build")
        self.lenet.build()

    def train(self):
        pass

    def classify(self):
        pass

class ImgPreprocess():
    def __init__(self):
        self.a = 1
    def imgStats(self):
        pass
    def preprocess(self):
        pass

def main():
    # set input training image directory
    # run pipeline
    # output classification results
    traffic_sign_pipeline = Pipeline()

if __name__ == '__main__':
    main()

