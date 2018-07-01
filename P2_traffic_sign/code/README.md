# **Traffic Sign Classifier Pipeline** 
-----
# aws:
change to an empty directory:
scp source dest

scp carnd@52.23.187.65:/home/carnd/UdacityP2/code/trafficSign.h5 ./

# reference:
adapted LeNet keras code from: https://github.com/f00-/mnist-lenet-keras/blob/master/lenet.py

adapted data augmentation idea from https://github.com/carlosgalvezp/Udacity-Self-Driving-Car-Nanodegree/blob/master/term1/projects/p2-traffic-signs/Traffic_Signs_Recognition.ipynb

http://jokla.me/robotics/traffic-signs/

Keras data genenerator: 

https://github.com/ndrplz/self-driving-car/blob/master/project_2_traffic_sign_classifier/Traffic_Sign_Classifier.ipynb

https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

Keras early stopping tutorial:
https://chrisalbon.com/deep_learning/keras/neural_network_early_stopping/

cited from early stopping tutorial:
we included EarlyStopping(monitor='val_loss', patience=2) to define that we wanted to monitor 
the test (validation) loss at each epoch and after the test loss has not improved after two epochs, 
training is interrupted. However, since we set patience=2, we won’t get the best model, 
but the model two epochs after the best model. 
Therefore, optionally, we can include a second operation, 
ModelCheckpoint which saves the model to a file after every checkpoint 
(which can be useful in case a multi-day training session is interrupted for some reason. Helpful for us, 
if we set save_best_only=True then ModelCheckpoint will only save the best model.

Keras load model for prediction tutorial:
https://github.com/EN10/KerasMNIST/blob/master/TFKpredict.py



# train.py:
### main():
step0: define hyper parameters, learning rate = 0.01  <br/>
step1: call prepareDataPipeline to get preprocessed data.  <br/>
step2: call run-training on training dataset, and printout accuracy on validation dataset  <br/>


### run-training: 
Use LeNet, inputs are 32x32x3 RGB images , gradient descent is done using Adam optimizer <br/>
During traning, use validation set every 10 episode to check the accracy.<br/>

### test: 
load saved model and test on test data set  <br/>
Please ignore this function for now, since it's not used during training. <br/>


# LeNet.py:

LeNet class defines the graph for a modified LeNet structure: <br/>
conv1 with relu: output (?, 32, 32, 6)    max pool:  (?, 16, 16, 6) <br/>
conv2 with relu: output (?, 16, 16, 16)   max pool:  (?, 8, 8, 16) <br/> 
conv3 with relu: output  (?, 8, 8, 16)    max pool:  (?, 4, 4, 16) <br/>
fc0 is the conv3 output flattened to (?, 256) <br/>
fc1   (?, 120)  <br/>
fc2   (?, 84)  <br/>
logits (?, num class)  <br/>
Cross entropy loss was used here.  <br/>

# tfUtil.py
Helper functions to build the graph in LeNet.py that's friendly to display graph and write summary for variables on tensorboard.  <br/>
conv-layer function creates a convolution layer with activation, maxpool and option for dropout. <br/>
fc-layer function creates an fc layer with activation if it's a hidden layer, without activaation if it's a logits layer.  <br/>

# dataUtil.py:
prepareDataPipeline loads the data and normalize the images. <br/>
normalizeAll normalizes images in test, train, and validation sets using X_data / 255. - 0.5   <br/>

Other functions that I implemented but commented out:  <br/>
dataAugmentation :  create data agumentaion with balanced number of examples per class. <br/>
visualize: this function displays example images for each class in the dataset , and plot a histogram for the distribution of data for each class.<br/>
preprocess: take Y channel from YUV as indicated in the recommended paper. The single channel images are later normalized  <br/>
testTF: test one hot encoding in tensorflow.   <br/>

# testTF.py
Please ignore this file, used for testing tensorflow functions. 

# Testing results:
1. Original setting , classic Lenet with 2 conv layers, 2 fc layers, Y channel from YUV image, normalize using X = (X - np.amin(X)) / (np.amax(X) - np.amin(X)), the accuracy is 88% after 20 epochs

2. If I use RGB (3 channels), normalize using (X_data / 255. - 0.5).astype(np.float32), 3 conv layers, 2 fc layer, I get accuracy 0.846 at 20 epochs, and 0.83582765 at 30 epochs

