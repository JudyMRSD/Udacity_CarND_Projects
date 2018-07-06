# This is NOT Jin's code!!!!!
# This is NOT Jin's code!!!!!
# This is NOT Jin's code!!!!!
# This is NOT Jin's code!!!!!
# This is NOT Jin's code!!!!!
# This is NOT Jin's code!!!!!

# example implementation in : https://github.com/mvpcom/Udacity-CarND-Project-3


ModelDir = "../data/model/"
Driving_Log_Path = "../data/driving_log.csv"
Img_Data_Dir = "../data/IMG/"

originalHyperparam = False

import pandas as pd
import os.path

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import matplotlib
# get_ipython().magic(u'matplotlib inline')
matplotlib.style.use('ggplot')
drivingLog = pd.read_csv(Driving_Log_Path,names=['Center','Left','Right','Steering Angle','Throttle','Break','Speed'],header=None)
drivingLog = drivingLog[1:]#remove heading
# In[3]:
# todo: remove anything for drivingLogTest
# load data
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# get_ipython().magic(u'matplotlib inline')
matplotlib.style.use('ggplot')
drivingLogTest = pd.read_csv(Driving_Log_Path,names=['Center','Left','Right','Steering Angle','Throttle','Break','Speed'],header=None)
drivingLogTest = drivingLogTest[1:]



# In[4]:

# load images
import cv2
import numpy as np
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
numSample = 50
# 1:numSample  do not include header ?
centerImgs = np.array([loadImg(imgLoc, True) for imgLoc in drivingLog['Center'][0:numSample]], dtype=np.float32)
leftImgs = np.array([loadImg(imgLoc, True) for imgLoc in drivingLog['Left'][0:numSample]], dtype=np.float32)
rightImgs = np.array([loadImg(imgLoc, True) for imgLoc in drivingLog['Right'][0:numSample]], dtype=np.float32)


# In[8]:



# In[5]:


# show all dataset to choose 
# source: http://stackoverflow.com/questions/21360361/how-to-dynamically-update-a-plot-in-a-loop-in-ipython-notebook-within-one-cell
import time
import pylab as pl
# from IPython import display


    
# results:     # filtering out dataset
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
# In[6]:

# data selection
drivingLog['Steering Angle'] = drivingLog['Steering Angle'].astype(float)
abs_angles = abs(drivingLog['Steering Angle'])
# main idea: https://github.com/budmitr # Dmitrii Budylskii
nonZeroSamples =  drivingLog.loc[abs_angles >0.01,:]
zeroSamples =  drivingLog.loc[abs_angles <0.01,:]  #
#print(nonZeroSamples)


# In[7]:

fractionVar = 0.25
newDrivingLog = pd.concat([nonZeroSamples, zeroSamples.sample(frac=fractionVar)], ignore_index=True)


# In[8]:

drivingLog = newDrivingLog


# In[9]:



# In[10]:

# Generator
def generateBatch(data, labels, batchSize=10, threshold=0.2):
    
    keepProbability = 0.0
    batchCount = len(labels)/batchSize 
    inputShape = loadImg(data['Center'][0], True) 
    batchXCenter = np.zeros((batchSize, inputShape.shape[0], inputShape.shape[1], inputShape.shape[2]))
    
    batchY = np.zeros(batchSize)  

    while True: # to make sure we never reach the end
        #if startIdx > len(data):
        #    startIdx = 0
        #endIdx = startIdx + batchSize
        #batchXCenter = np.array([loadImg(imgLoc) for imgLoc in data['Center'][startIdx:endIdx]], dtype=np.float32)
        #batchXLeft = np.array([loadImg(imgLoc) for imgLoc in data['Left'][startIdx:endIdx]], dtype=np.float32)
        #batchXRight = np.array([loadImg(imgLoc) for imgLoc in data['Right'][startIdx:endIdx]], dtype=np.float32) 
        #yield batchXCenter, batchXLeft, batchXRight, labels[startIdx:endIdx], startIdx
        #for i in range(startIdx,endIdx):
        counter = 0
        while counter<=batchSize-1:
            # print("counter", counter)
            idx = np.random.randint(len(labels)-1) 
            steeringAngle = labels[idx]

            imgLoc = data['Center'][idx]

            image = loadImg(imgLoc, True) 

            # randomly augment data
            if np.random.rand() > 0.5: # 50 percent chance to see the right angle
                image = cv2.flip(image,1)
                steeringAngle = -steeringAngle

            rows, cols, _ = image.shape
            transRange = 100
            numPixels = 10
            valPixels = 0.4
            transX = transRange * np.random.uniform() - transRange/2
            steeringAngle = steeringAngle + transX/transRange * 2 * valPixels
            transY = numPixels * np.random.uniform() - numPixels/2
            transMat = np.float32([[1,0, transX], [0,1, transY]])
            image = cv2.warpAffine(image, transMat, (cols, rows))
                
            if np.random.rand() <= 1.0: # always # augment image's light and brightnesss
                # http://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #convert it to hsv
                randomLight = 0.25 + np.random.rand() 
                hsv[:,:,2] =  hsv[:,:,2] * randomLight
                newImage = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                
            # sanity check of data
            image = image.reshape(inputShape.shape[0], inputShape.shape[1], inputShape.shape[2])
            if(image.shape[0]==0):
                continue
            
            batchXCenter[counter] = image
            batchY[counter] = steeringAngle
            counter += 1
        
        yield batchXCenter, batchY
    
def generateBatchVal(data, labels, batchSize=10):
    startIdx = 0
    batchCount = len(labels)/batchSize 
    while True: # to make sure we never reach the end
        endIdx = startIdx + batchSize
        batchXCenter = []
        for imgLoc in data['Center'][startIdx:endIdx]:
            imgLoc = Img_Data_Dir+ imgLoc.split('/')[-1]
            if (os.path.isfile(imgLoc)):
                batchXCenter.append(loadImg(imgLoc, False))
            else:
                print("image not exist:", imgLoc)
        batchXCenter = np.array(batchXCenter, dtype=np.float32)
        #batchXCenter = np.array([loadImg(imgLoc, False) for imgLoc in data['Center'][startIdx:endIdx]], dtype=np.float32)
        
        #batchXLeft = np.array([loadImg(imgLoc) for imgLoc in data['Left'][startIdx:endIdx]], dtype=np.float32)
        #batchXRight = np.array([loadImg(imgLoc) for imgLoc in data['Right'][startIdx:endIdx]], dtype=np.float32) 
        #yield batchXCenter, batchXLeft, batchXRight, labels[startIdx:endIdx], startIdx
        batchY = labels[startIdx:endIdx]
        yield batchXCenter, batchY
        startIdx = endIdx
        if startIdx > len(data)-1:
            startIdx = 0





# In[9]:

#drivingLog = drivingLog.drop('level_0', 1)

from sklearn.model_selection import train_test_split
XTrain, XVal, yTrain, yVal = train_test_split(drivingLog,drivingLog['Steering Angle'],test_size=0.20,random_state=0)
# In[16]:

XTrain = drivingLog
yTrain = drivingLog['Steering Angle']
XVal = drivingLogTest
yVal = drivingLogTest['Steering Angle']


# In[17]:



XTrain = XTrain.reset_index(drop=True)
XVal = XVal.reset_index(drop=True)
# yTrain = yTrain.reset_index(drop=True)
# yTrain = yTrain['Steering Angle']
# yVal = yVal.reset_index(drop=True)
# yVal = yVal['Steering Angle']
# In[20]:

# augment data to a dilstribution same to normal distribution
# main idea: Vivek Yadav (https://github.com/vxy10/) 
idx = 100
test = XTrain['Center'][idx]
steeringAngle = yTrain[idx]
image = loadImg(test, True) 
plt.figure()
if image.shape[2]<3:
    plt.imshow(image.reshape(image.shape[0],image.shape[1]), cmap=plt.get_cmap('gray'))
else:
    plt.imshow(image)

rows, cols, _ = image.shape
transRange = 150
numPixels = 10
valPixels = 0.2
transX = transRange * np.random.uniform() - transRange/2
newsteeringAngle = steeringAngle + transX/transRange * 2 * valPixels
transY = numPixels * np.random.uniform() - numPixels/2
transMat = np.float32([[1,0, transX], [0,1, transY]])
newImage = cv2.warpAffine(image, transMat, (cols, rows))

#img = img[60:-20,:,:]
#newImage = cv2.resize(newImage, (64, 64), interpolation=cv2.INTER_AREA)
    
rows,cols,dim3 = newImage.shape
plt.figure()
if dim3<3:
    plt.imshow(newImage.reshape(newImage.shape[0],newImage.shape[1]), cmap=plt.get_cmap('gray'))
else:
    plt.imshow(newImage)
            


# In[130]:

# augment light and brightness
idx = 2000
test = XTrain['Center'][idx]
testLabel = yTrain[idx]
image = loadImg(test, True) 
#image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
plt.figure()
if image.shape[2]<3:
    plt.imshow(image.reshape(image.shape[0],image.shape[1]), cmap=plt.get_cmap('gray'))
else:
    plt.imshow(image)

hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #convert it to hsv
#h, s, v = cv2.split(hsv)
randomLight = 0.25 + np.random.rand() 
hsv[:,:,2] =  hsv[:,:,2] * randomLight
newImage = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB) 
rows,cols,dim3 = newImage.shape
plt.figure()
if dim3<3:
    plt.imshow(newImage.reshape(newImage.shape[0],newImage.shape[1]), cmap=plt.get_cmap('gray'))
else:
    plt.imshow(newImage)


# In[27]:

## test left and right augmentation
# augment test
idx = 2000
test = XTrain['Center'][idx]
testLabel = yTrain[idx]
image = loadImg(test, True) 
plt.figure()
if image.shape[2]<3:
    plt.imshow(image.reshape(image.shape[0],image.shape[1]), cmap=plt.get_cmap('gray'))
else:
    plt.imshow(image)

newImage = XTrain['Left'][idx]
newImage = loadImg(newImage, True) 
rows,cols,dim3 = newImage.shape
transMat = np.float32([[1,0,-15],[0,1,0]])
newImage = cv2.warpAffine(newImage,transMat,(cols,rows))
plt.figure()
if dim3<3:
    plt.imshow(newImage.reshape(newImage.shape[0],newImage.shape[1]), cmap=plt.get_cmap('gray'))
else:
    plt.imshow(newImage)

#test right
newImage = XTrain['Right'][idx]
newImage = loadImg(newImage, True) 
rows,cols,dim3 = newImage.shape
transMat = np.float32([[1,0,5],[0,1,0]])
newImage = cv2.warpAffine(newImage,transMat,(cols,rows))
plt.figure()
if dim3<3:
    plt.imshow(newImage.reshape(newImage.shape[0],newImage.shape[1]), cmap=plt.get_cmap('gray'))
else:
    plt.imshow(newImage)



# In[25]:

# augment test
idx = 1000
test = XTrain['Center'][idx]
testLabel = yTrain[idx]
image = loadImg(test, True) 
plt.figure()
if image.shape[2]<3:
    plt.imshow(image.reshape(image.shape[0],image.shape[1]), cmap=plt.get_cmap('gray'))
else:
    plt.imshow(image)

newImage = cv2.flip(image,1)
plt.figure()
plt.imshow(newImage)



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

# config
#inputShape = (110,320,3) 
#inputShape = (55,160,3) # centerImgs.shape # half for cpu
#inputShape = (128,128,3) # centerImgs.shape # half for cpu
inputShape = (40,160,3)

# my architecture 
model = Sequential()
model.add(Lambda(lambda x: x/255.-0.5,input_shape=inputShape))
model.add(Conv2D(3,1,1, activation = 'elu', name='Conv2D1')) # color space 

model.add(Conv2D(16,8,8, activation = 'elu', name='Conv2D2'))
#model.add(Conv2D(16,9,9, activation = 'elu', name='Conv2D2'))
model.add(MaxPooling2D(pool_size=(2,2), name='MaxPoolC2'))
model.add(Dropout(0.3, name='DropoutC2'))

model.add(Conv2D(32,5,5, activation = 'elu', name='Conv2D3'))
#model.add(Conv2D(32,7,7, activation = 'elu', name='Conv2D3'))
model.add(MaxPooling2D(pool_size=(2,2), name='MaxPoolC3'))
model.add(Dropout(0.3, name='DropoutC3'))

model.add(Conv2D(32,3,3, activation = 'elu', name='Conv2D4'))
#model.add(Conv2D(128,3,3, activation = 'elu', name='Conv2D5'))
model.add(MaxPooling2D(pool_size=(2,2), name='MaxPoolC4'))
model.add(Dropout(0.3, name='DropoutC4'))

#model.add(Conv2D(64,3,3, activation = 'elu', name='Conv2D5'))
#model.add(Conv2D(128,3,3, activation = 'elu', name='Conv2D6'))
#model.add(MaxPooling2D(pool_size=(2,2), name='MaxPoolC6'))
#model.add(Dropout(0.5, name='DropoutC6'))

#model.add(Conv2D(256,2,2, activation = 'relu', name='Conv2D7'))
#model.add(MaxPooling2D(pool_size=(2,2), name='MaxPoolC7'))
#model.add(Dropout(0.5, name='DropoutC7'))

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

model.add(Dense(1,activation='elu', name='Output')) # problem is a regression


# In[27]:

# model summary 
model.summary()


# In[28]:

#adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer="adam", loss="mse")

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback, Callback

numTimes = 10
numEpoch = 10 # 4 + 2

thr = 0.0001 # 0.3
for time in range(numTimes):
    trainGenerator = generateBatch(XTrain, yTrain, batchSize=50, threshold=thr)
    validGenerator = generateBatchVal(XVal, yVal, batchSize=50)
    samplesPerEpoch = 32000 
    nbValSamples = 1000
    #history = model.fit_generator(trainGenerator, samplesPerEpoch, numEpoch, 
    #                verbose=1, validation_data=validGenerator, nb_val_samples = nbValSamples,
     #               callbacks=[ModelCheckpoint(filepath="bestVal.h5", verbose=1, save_best_only=True), ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=0.000001)])
    history = model.fit_generator(trainGenerator, samples_per_epoch=samplesPerEpoch, nb_epoch=numEpoch, validation_data=validGenerator,
                    nb_val_samples=nbValSamples, callbacks=[ModelCheckpoint(filepath=ModelDir+"bestVal.h5", verbose=1, save_best_only=True)]
                )
    print(thr, 'Time ',time+1)
    thr += (1/(numTimes))
