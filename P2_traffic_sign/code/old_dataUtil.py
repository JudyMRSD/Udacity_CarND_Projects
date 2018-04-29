import cv2
import pickle

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from skimage import exposure
import tensorflow as tf

from collections import defaultdict


def prepareDataPipeline():
    # Step 1: Import data

    X_train_coord, X_train, y_train, X_valid, y_valid, X_test, y_test = loadData()
    # X_train, X_test, X_valid = preprocess_all(X_train, X_test, X_valid)
    # X_test, X_train, X_valid = normalizeAll(X_test, X_train, X_valid)
    #visualize(X_train, y_train, imgPath='../writeup/visualizeData')
    
    # Step 2: Use data agumentation to make more training data
    
    X_train_new, y_train_new = dataAugmentation(X_train, y_train)
    
    #print("before augment: number of training data  = ", X_train.shape[0])
    X_train = np.concatenate((X_train, X_train_new), axis=0)
    y_train = np.concatenate((y_train, y_train_new), axis=0)

    #print("after augment: number of training data  = ",X_train.shape[0])
    #visualize(X_train, y_train, imgPath='../writeup/visualizeAugment')

    # Step 3: Data processing for tarin, validation, and test dataset
    X_train, X_valid, X_test = preprocess_gray(X_train, X_valid, X_test)
    # X_train, y_train, X_valid, y_valid, X_test, y_test = preprocess(X_train, y_train, X_valid, y_valid, X_test, y_test)
    # Step 4: visualize preprocessed data

    visualize(X_train, y_train, imgPath='../writeup/visualizeData-ychannel', isGray=True)

    return X_train, y_train, X_valid, y_valid, X_test, y_test



def evaluate(X_data, y_data, BATCH_SIZE, accuracy_operation):

    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


# load pickled data

def loadData():
    data_folder = "../traffic-signs-data/"
    training_file = data_folder + "train.p"
    validation_file = data_folder + "valid.p"
    testing_file = data_folder + "test.p"

    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    with open(testing_file, mode='rb') as f:
        test = pickle.load(f)
    X_train_coord = train['coords'];
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    X_test, y_test = test['features'], test['labels']

    assert (len(X_train) == len(y_train))
    assert (len(X_valid) == len(y_valid))
    assert (len(X_test) == len(y_test))


    return X_train_coord, X_train, y_train, X_valid, y_valid, X_test, y_test

# Visualizations for distribution of data and image example for each class
def visualize(X, y, imgPath, isGray=False):
    freq_all = defaultdict(int)
    for c in y:
        freq_all[c] += 1


    plt.close('all')
    # classes is the ordered unique classes

    # indices:  indices of input array that result in the unique array classes
    # index of examples that is the first occurence of a sign,
    # the first 40 will be the first 40 classes

    # counts: the number of times each unique sign appears

    classes, indices, counts = np.unique(y, return_index=True, return_counts=True)
    print("list of classes", classes)  # [0,...42]
    num_classes = len(classes)
    print("number of classes", num_classes)  # 43

    # plotting the count of each sign
    # historgram bins arranged by classes
    print("historgram bins arranged by classes ")


    numBins = int(classes.shape[0])
    plt.hist(y,bins=numBins)

    plt.title("Training Data Histogram")
    plt.xlabel("Class")
    plt.ylabel("Occurence")
    plt.savefig(imgPath+'_histogram.jpg')
    #plt.show()

    # unique_images are the first occurence of traffic signs in the data set
    unique_images = X[indices]
    print("num unique images : ", len(unique_images))


    fig = plt.figure()
    for i in range(num_classes):
        ax = fig.add_subplot(5, 9, i + 1, xticks=[], yticks=[])
        ax.set_title(i)
        if (isGray == True):
            ax.imshow(np.squeeze(unique_images[i]), cmap='gray')

        else:
            ax.imshow(unique_images[i])

    plt.savefig(imgPath+'_sample')
    plt.close('all')
# opencv documentation
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
# adapted idea from:
# https://github.com/carlosgalvezp/Udacity-Self-Driving-Car-Nanodegree/blob/master/term1/projects/p2-traffic-signs/Traffic_Signs_Recognition.ipynb
def traslation(src, width_shift_range = 0.1, height_shift_range = 0.1):
    height = src.shape[0]
    width = src.shape[1]
    tx = np.random.uniform(-width*width_shift_range, width*width_shift_range)
    ty = np.random.uniform(-height*height_shift_range, height*height_shift_range)
    M = np.array([[1, 0, tx], [0, 1, ty]], dtype = np.float32)
    dst = cv2.warpAffine(src, M, (width, height))
    return dst

def rotation(src, rotation_rage = 15):
    height = src.shape[0]
    width = src.shape[1]
    theta = np.random.uniform(-rotation_rage, rotation_rage)
    M = cv2.getRotationMatrix2D((width / 2, height / 2), theta, 1)
    dst = cv2.warpAffine(src, M, (width, height))
    return dst

def scale(src, zoom_range= 0.2):
    height = src.shape[0]
    width = src.shape[1]
    x_scale = np.random.uniform(-zoom_range, zoom_range) + 1
    y_scale = np.random.uniform(-zoom_range, zoom_range) + 1
    M = np.array([[x_scale, 0, 0],[0, y_scale, 0]])
    dst = cv2.warpAffine(src, M, (width, height))

    return dst

def randomTransform(src,
                    width_shift_range=0.1, height_shift_range=0.1,
                    rotation_rage = 15,
                    zoom_range = 0.2):
    transformId = np.random.randint(0,3)
    if transformId == 0:
        dst = traslation(src, width_shift_range, height_shift_range)
        #cv2.imshow("scale", dst)
    elif transformId == 1:
        dst = rotation(src, rotation_rage)
        #cv2.imshow("rotate", dst)
    elif transformId == 2:
        dst = scale(src, zoom_range)
        #cv2.imshow("zoom", dst)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return dst

# factor by which to expand data
def dataAugmentation(X_train, y_train, factor = 5):
    print("enter data augmentation")
    # count frequency of each class
    freq = defaultdict(int)
    for c in y_train:
        freq[c] += 1

    # target count per class
    originalNumImg = y_train.shape[0]
    numClass = len(freq)

    final_count = originalNumImg * factor

    count_per_class = final_count/numClass
    # number of fake data per old image for each class
    fake_freq = defaultdict(int)
    # e.g.   old distribution  70  vs 30
    # target   200  vs 200
    # new data needed to balance:
    # (200 - 70) /70 = ceil(140 / 70) = 2
    # (200 - 30) /30 = ceil(170 / 30) = 6

    # (count_per_class - old_count_of_the_class) / old_count_of_the_class
    for k,v in freq.items():
        fake_freq[k] = int(np.ceil( ( count_per_class - v)/v))


    X_train_new = []
    y_train_new = []
    # balance data distribution
    for i in range (originalNumImg):
        # number of images needed to blalance distribution among classes
        n_augment = fake_freq[ y_train[i] ]
        # create agumented images
        for j in range (n_augment):
            newImg = randomTransform(X_train[i])
            X_train_new.append(newImg)
            y_train_new.append(y_train[i])
        # visualize the first augmentation
        if (i==0):
            visualize_single_augment(X_train_new)

    X_train_new = np.array(X_train_new)
    y_train_new = np.array(y_train_new)
    print(y_train_new.shape[0])



    return X_train_new, y_train_new

def visualize_single_augment(X_new):
    fig = plt.figure()
    for i in range(len(X_new)):
        ax = fig.add_subplot(5, 9, i + 1, xticks=[], yticks=[])
        ax.set_title(i)
        ax.imshow(np.squeeze(X_new[i]), cmap='gray')

    plt.savefig('../writeup/visualizeAugment_singleImg.jpg')
    plt.close('all')


def Y_channel_YUV(X):
    # Y channel calculation from: https://github.com/navoshta/traffic-signs/blob/master/Traffic_Signs_Recognition.ipynb
  
    threeChannelShape = X.shape
    # shape is tuple, not mutable
    singleChannelShape = threeChannelShape[0:3] + (1,)
    # set to single channel
    X_singleChannel = np.zeros(singleChannelShape)

    for i in range(0, len(X)):
        img = X[i]
        yuv_img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        y, u, v = cv2.split(yuv_img)
        y = np.expand_dims(y, axis=2)
        X_singleChannel[i] = y

    # plt.imshow(X[0], cmap='gray')
    # plt.show()
 
    return X

def gray(X):
    # Y channel calculation from: https://github.com/navoshta/traffic-signs/blob/master/Traffic_Signs_Recognition.ipynb
    threeChannelShape = X.shape
    # shape is tuple, not mutable
    singleChannelShape = threeChannelShape[0:3] + (1,)
    # set to single channel
    X_singleChannel = np.zeros(singleChannelShape)

    for i in range(0, len(X)):
        img = X[i]
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # print("gray shape", gray_img.shape) # (32, 32)
        gray_img = np.expand_dims(gray_img, axis=2)
        # TODO: fix this
        gray_img=normalization(gray_img)

        X_singleChannel[i] = gray_img

        # print("gray img shape",X_singleChannel[i].shape) # (32,32,1)

    # plt.imshow(X[0], cmap='gray')
    # plt.show()
 
    return X_singleChannel

def normalization(x):
    x_min = 0
    x_max = 255
    return (x - x_min) / (x_max - x_min)


def preprocess_gray(X_train, X_valid, X_test):
     
    # normalized gray images
    X_train = gray(X_train)
    X_valid = gray(X_valid)
    X_test = gray(X_test)
        
    return X_train, X_valid, X_test


def testTF():
    x = tf.placeholder(tf.float32, (100, 32, 32, 1))
    y = tf.placeholder(tf.int32, (100))
    y_hot = tf.placeholder(tf.int32, (100, 43))
    one_hot_y = tf.one_hot(y, 43)

    print(y_hot.get_shape())  # (100, 43)
    print("tf.one_hot=", one_hot_y.get_shape())  # (100, 43)