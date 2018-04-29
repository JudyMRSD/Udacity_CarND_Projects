from LeNet import LeNet

class TrafficSignClassifier():
    num_classes = 40
    img_width, img_height, img_channels = [32, 32, 1]
    img_params = [img_width, img_height, img_channels]
    # img_params = img.shape()
    lenet = LeNet(num_classes, img_params)
    lenet.build()

tsc = TrafficSignClassifier()
