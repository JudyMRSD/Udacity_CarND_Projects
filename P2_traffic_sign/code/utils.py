import pickle

import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
plt.switch_backend('agg')


class TrainMonitorTools():
    def __init__(self):
        print("train process tool")
    def visualizeTrain(self, visualize_dir, history):
        print(history.history.keys())  # dict_keys(['val_acc', 'acc', 'loss', 'val_loss'])
        plt.close('all')
        # summarize history for accuracy
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(visualize_dir+'acc.jpg')
        plt.show()
        # summarize history for loss
        plt.close('all')

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        plt.savefig(visualize_dir+'loss.jpg')

class DataSetTools():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.ground_truth_dir = self.data_dir + 'dataset_groundtruth/'
        self.visualize_dir=self.data_dir+'visualize/'

    def loadData(self):
        print("load trianing and validation data")
        training_file = self.ground_truth_dir + "train.p"
        validation_file = self.ground_truth_dir + "valid.p"
        testing_file = self.ground_truth_dir + "test.p"

        with open(training_file, mode='rb') as f:
            train = pickle.load(f)
        with open(validation_file, mode='rb') as f:
            valid = pickle.load(f)
        with open(testing_file, mode='rb') as f:
            test = pickle.load(f)

        self.X_train, self.y_train = train['features'], train['labels']
        self.X_valid, self.y_valid = valid['features'], valid['labels']
        self.X_test, self.y_test = test['features'], test['labels']


    def summarizeData(self):
        self.n_train = self.X_train.shape[0]
        self.n_test = self.X_test.shape[0]
        self.image_shape = self.X_train[0].shape

        self.n_classes = np.unique(self.y_train).shape[0]

        print("Number of training examples =", self.n_train)
        print("Number of testing examples =", self.n_test)
        print("Image data shape =", self.image_shape)
        print("Number of classes =", self.n_classes)

    def visualizeHistogram(self, labels, tag):
        # histogram of classes
        plt.close('all')
        plt.hist(labels, bins=self.n_classes)
        plt.title(tag + "Data Histogram")
        plt.xlabel("Class")
        plt.ylabel("Occurence")
        plt.savefig(self.visualize_dir + tag + '_histogram.jpg')
        plt.show()

    def visualizeUniqueImgs(self, labels, imgs, tag, numImgs, numRows, data_generator = None):
        # plot unique images
        _, unique_indices = np.unique(labels, return_index=True)
        unique_images = imgs[unique_indices]

        if data_generator:
            unique_labels = np.arange(len(unique_indices))+1
            unique_images, label = data_generator.flow(unique_images, unique_labels, len(unique_indices)).next() # (43, 32, 32, 3)
            unique_images = np.array(unique_images, dtype=np.uint8) # change to uint8 for plotting

        # plot images
        fig = plt.figure()
        for i in range(numImgs):
            ax = fig.add_subplot(numRows, len(unique_indices) / numRows + 1, i + 1, xticks=[], yticks=[])
            ax.set_title(unique_indices[i])
            ax.imshow(unique_images[i])

        plt.savefig(self.visualize_dir + tag + '_sample.jpg')
        plt.show()
        plt.close('all')

    def data_augment(self):
        # balance using keras ImageDataGenerator
        self.data_generator = ImageDataGenerator(
                            featurewise_center=True,
                            data_format='channels_last')
        # call fit to obtain mean value of the dataset to use in featurewise_center
        self.data_generator.fit(self.X_train)


    def load_explore(self):
        self.loadData()
        self.summarizeData()
        print("visualize before data augmentation on training data")
        self.visualizeUniqueImgs(self.y_train, self.X_train, tag='train', numImgs = self.n_classes, numRows = 5)
        print("visualize training data distribution over class")
        self.visualizeHistogram(self.y_train, tag='train')

        self.data_augment()
        self.visualizeUniqueImgs(self.y_train, self.X_train, tag='train_aug',
                                 numImgs = self.n_classes, numRows = 5, data_generator=self.data_generator)
        # summarize dataset info
        img_width, img_height, img_channels = self.image_shape
        print("image shape: img_width = {}, img_height = {}, img_channels = {}".format(img_width, img_height, img_channels))
        return self.n_classes, img_width, img_height, img_channels


    def top_k(self, pred_prob, test_labels, k = 5):
        out_class = []
        for i in range(0, len(pred_prob)):
            print("Top 5 probability for image id = ", i)
            p = pred_prob[i, :]
            # [::-1] reverse the array,  sort from low to high
            top_class = np.argsort(p)[::-1][:k]# top k
            out_class.append(top_class[0])
            top_prob = p[top_class]
            for j in range(0, k):
                print("class: {}, prob:{}".format(top_class[j], top_prob[j]))
        accuracy = np.sum(test_labels == out_class) / len(test_labels)
        print("Ground truth: ", test_labels)
        print("Predicted labels", out_class)
        print("Accuracy on testing images =", accuracy)