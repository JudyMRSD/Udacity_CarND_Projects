# This is NOT Jin's code!!!!!
# This is NOT Jin's code!!!!!
# This is NOT Jin's code!!!!!
# This is NOT Jin's code!!!!!
# This is NOT Jin's code!!!!!
# This is NOT Jin's code!!!!!

# example implementation in : https://github.com/mvpcom/Udacity-CarND-Project-3

# try different settings
# original setting gpu2
# Expand_Log = True
# Less_Zero = True

# turn both off gpu 1
Expand_Log = False
Less_Zero = False

Param_Name = ""
if Expand_Log:
    Param_Name+="El"
if Less_Zero:
    Param_Name+="Lz"


# TODO: check the "different" tag
ModelDir = "../data/model/"
Driving_Log_Path = "../data/driving_log.csv"
Img_Data_Dir = "../data/IMG/"

originalHyperparam = False

import os.path

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, Callback


# In[26]:

# model design
import pickle
import numpy as np
import math
from keras.utils import np_utils
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Dense, Activation
from keras.optimizers import SGD, Adam
from keras.models import Sequential

drivingLog = pd.read_csv(Driving_Log_Path,names=['Center','Left','Right','Steering Angle','Throttle','Break','Speed'],header=None)
drivingLog = drivingLog[1:]#remove heading
drivingLogTest = pd.read_csv(Driving_Log_Path,names=['Center','Left','Right','Steering Angle','Throttle','Break','Speed'],header=None)
drivingLogTest = drivingLogTest[1:]


def loadImg(imgLoc,trainFlag):
    imgLoc = imgLoc.split('/')[-1]

    if trainFlag:
        imageLocation = Img_Data_Dir+imgLoc
        image = cv2.imread(imageLocation,cv2.IMREAD_COLOR) # BGR
    else:
        imageLocation = Img_Data_Dir+imgLoc
        image = cv2.imread(imageLocation,cv2.IMREAD_COLOR)
    if (image is None):
        print("image is None",imageLocation)

    image = image[60:-20,:,:] # vivek approach
    image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Generator
def generateBatch(data, labels, batchSize=10,):
    inputShape = loadImg(data['Center'][0], True)
    batchXCenter = np.zeros((batchSize, inputShape.shape[0], inputShape.shape[1], inputShape.shape[2]))

    batchY = np.zeros(batchSize)

    while True:  # to make sure we never reach the end
        counter = 0
        while counter <= batchSize - 1:
            # print("counter", counter)
            idx = np.random.randint(len(labels) - 1) + 1 # + 1 to make sure not taking 0th index
            if (idx>len(labels)-1):
                print("idx=",idx, " len(labels)-1=", len(labels)-1)
            #print("idx", idx)
            steeringAngle = labels[idx]
            # print("steeringAngle", steeringAngle)

            imgLoc = data['Center'][idx]
            image = loadImg(imgLoc, True)

            # randomly augment data
            if np.random.rand() > 0.5:  # 50 percent chance to see the right angle
                image = cv2.flip(image, 1)
                steeringAngle = -steeringAngle

            rows, cols, _ = image.shape
            transRange = 100
            numPixels = 10
            valPixels = 0.4
            transX = transRange * np.random.uniform() - transRange / 2
            steeringAngle = steeringAngle + transX / transRange * 2 * valPixels
            transY = numPixels * np.random.uniform() - numPixels / 2
            transMat = np.float32([[1, 0, transX], [0, 1, transY]])
            image = cv2.warpAffine(image, transMat, (cols, rows))

            if np.random.rand() <= 1.0:  # always # augment image's light and brightnesss
                # http://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # convert it to hsv
                randomLight = 0.25 + np.random.rand()
                hsv[:, :, 2] = hsv[:, :, 2] * randomLight
            # sanity check of data
            image = image.reshape(inputShape.shape[0], inputShape.shape[1], inputShape.shape[2])
            if (image.shape[0] == 0):
                continue

            batchXCenter[counter] = image
            batchY[counter] = steeringAngle
            counter += 1

        yield batchXCenter, batchY


def generateBatchVal(data, labels, batchSize=10):
    startIdx = 0

    while True:  # to make sure we never reach the end
        endIdx = startIdx + batchSize
        batchXCenter = []
        for imgLoc in data['Center'][startIdx:endIdx]:
            imgLoc = Img_Data_Dir + imgLoc.split('/')[-1]
            if (os.path.isfile(imgLoc)):
                batchXCenter.append(loadImg(imgLoc, False))
            else:
                print("image not exist:", imgLoc)
        batchXCenter = np.array(batchXCenter, dtype=np.float32)
        batchY = labels[startIdx:endIdx]
        yield batchXCenter, batchY
        startIdx = endIdx
        if startIdx > len(data) - 1:
            startIdx = 0

drivingLog['Steering Angle'] = drivingLog['Steering Angle'].astype(float)

if (Expand_Log):
    allIdx = list(range(0,8060)) + list(range(8140,8300)) + list(range(10100,10600)) + list(range(10720,13460)) + list(range(14940,15360)) + list(range(15600,15860)) + list(range(16240,16480)) + list(range(16980,17180)) + list(range(17640,18160)) + list(range(18460,19740)) + list(range(20120,20480)) + list(range(20600,21100)) + list(range(21240,21742))
    print("drivingLog shape",drivingLog.shape)
    newDrivingLog = drivingLog.ix[allIdx]
    #print(newDrivingLog['Steering Angle'])# preprocess data and augment data if necessary
    drivingLog = newDrivingLogdrivingLog = drivingLog.reset_index(drop=True)# select data from selected data
    allIdx = list(range(0,10000))
    newDrivingLog = drivingLog.ix[allIdx]
    drivingLog = newDrivingLog
    print("drivingLog after",drivingLog.shape)
    drivingLog = drivingLog.reset_index(drop=True)

if (Less_Zero):
    # data selection
    abs_angles = abs(drivingLog['Steering Angle'])
    # main idea: https://github.com/budmitr # Dmitrii Budylskii
    nonZeroSamples =  drivingLog.loc[abs_angles >0.01,:]
    zeroSamples =  drivingLog.loc[abs_angles <0.01,:]
    fractionVar = 0.25
    newDrivingLog = pd.concat([nonZeroSamples, zeroSamples.sample(frac=fractionVar)], ignore_index=True)
    drivingLog = newDrivingLog

# split train ,val
XTrain, XVal, yTrain, yVal = train_test_split(drivingLog,drivingLog['Steering Angle'],test_size=0.20,random_state=0)
print("drivingLog[0]", drivingLog['Steering Angle'][:2])
XTrain = drivingLog
yTrain = drivingLog['Steering Angle']
XVal = drivingLogTest
yVal = drivingLogTest['Steering Angle']

XTrain = XTrain.reset_index(drop=True)
XVal = XVal.reset_index(drop=True)


inputShape = (40,160,3)

# my architecture 
model = Sequential()
model.add(Lambda(lambda x: x/255.-0.5,input_shape=inputShape))
model.add(Conv2D(3,1,1, activation = 'elu', name='Conv2D1')) # color space 

model.add(Conv2D(16,8,8, activation = 'elu', name='Conv2D2'))
model.add(MaxPooling2D(pool_size=(2,2), name='MaxPoolC2'))
model.add(Dropout(0.3, name='DropoutC2'))

model.add(Conv2D(32,5,5, activation = 'elu', name='Conv2D3'))
model.add(MaxPooling2D(pool_size=(2,2), name='MaxPoolC3'))
model.add(Dropout(0.3, name='DropoutC3'))

model.add(Conv2D(32,3,3, activation = 'elu', name='Conv2D4'))
model.add(MaxPooling2D(pool_size=(2,2), name='MaxPoolC4'))
model.add(Dropout(0.3, name='DropoutC4'))

# convolution to dense
model.add(Flatten(name='Conv2Dense'))

model.add(Dense(256,activation='elu', name='Dense1'))
model.add(Dropout(0.5, name='DropoutD1'))

model.add(Dense(128,activation='elu', name='Dense2'))
model.add(Dropout(0.5, name='DropoutD2'))

model.add(Dense(64,activation='elu', name='Dense3'))
model.add(Dropout(0.5, name='DropoutD3'))

model.add(Dense(8,activation='elu', name='Dense4'))
model.add(Dropout(0.5, name='DropoutD4'))

model.add(Dense(1,activation='elu', name='Output'))

# model summary 
model.summary()
model.compile(optimizer="adam", loss="mse")


numEpoch = 17
trainGenerator = generateBatch(XTrain, yTrain, batchSize=50)
validGenerator = generateBatchVal(XVal, yVal, batchSize=50)
samplesPerEpoch = 32000
nbValSamples = 1000
history = model.fit_generator(trainGenerator, samples_per_epoch=samplesPerEpoch, nb_epoch=numEpoch, validation_data=validGenerator,
                nb_val_samples=nbValSamples, callbacks=[ModelCheckpoint(filepath=ModelDir+Param_Name+"July5.h5", verbose=1, save_best_only=True)]
            )