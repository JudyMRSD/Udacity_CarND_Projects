# Traffic Sign Classifier Pipeline
-----
## Overview
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

## File Structure
Run the piepeline using either `traffic_sign_classifier.py` or `Traffic_Sign_Classifier.ipynb`
Since I was training on GPU and ipynb isn't very friendly with that, I used traffic_sign_classifier.py 
instead. As a result, the training process wasn't recorded in ipynb and HTML file. 

```
project
│   writeup_report.md
└───code
│   |   utils.py                        Utility function to visualize data and monitor training process 
│   |   LeNet.py                        Constructs LeNet architecture 
|   |   traffic_sign_classifier.py      Main pipeline for training and testing traffic sign classifier 
│   |   Traffic_Sign_Classifier.ipynb   notebook file as main function 
└───data 
|   └───dataset_groundtruth/         Dataset for trianing, validation and testing    
|   └───env/                         Anaconda environment.yml files for MacOS and Ubuntu 
|   └───testImg/                     Traffic sign test images
│   └───model/                       
|       └─── trafficSign_model.p     Keras model for traffic sign classification 
|   └───visualize/                   *.jpg images for visualization
```

## Reference:
Adapted LeNet keras code from: </br>
`https://github.com/f00-/mnist-lenet-keras/blob/master/lenet.py`</br>
Adapted data augmentation idea from: </br>
`https://github.com/carlosgalvezp/Udacity-Self-Driving-Car-Nanodegree/blob/master/term1/projects/p2-traffic-signs/Traffic_Signs_Recognition.ipynb`
`http://jokla.me/robotics/traffic-signs/`</br>
Keras data genenerator: </br>
`https://github.com/ndrplz/self-driving-car/blob/master/project_2_traffic_sign_classifier/Traffic_Sign_Classifier.ipynb`
`https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html`

Keras load model for prediction tutorial: </br>
`https://github.com/EN10/KerasMNIST/blob/master/TFKpredict.py`</br>

Keras early stopping tutorial:</br>
`https://chrisalbon.com/deep_learning/keras/neural_network_early_stopping/`</br>

Testing images from :</br>
https://github.com/mohamedameen93/German-Traffic-Sign-Classification-Using-TensorFlow
## Data Set Summary & Exploration
Implementation is class DataSetTools() in utils.py</br>
Visualization and summary are shown in Traffic_Sign_Classifier.ipynb </br>

## Design and Test a Model Architecture
|experiment id|Validation Accuracy	| Image Augmentation	|  Network Architecture  |
|----|-------|-------------------------------------|------------------------------|
|0 | 0.8	| gray, r=15, w=0.1, h=0.1, z=0.2, hf, sn|c(6,5,relu), c(16, 5, relu), c(16, 5, relu), F, d(256, relu)|
|1  |0.64 	| **color**, r=15, w=0.1, h=0.1, z=0.2, hf, sc|c(6,5,relu), c(16, 5, relu), c(16, 5, relu), F, d(256, relu)|
|2 |0.78	| color, **sc** |c(6,5,relu), c(16, 5, relu), c(16, 5, relu), F, d(256, relu) |
|3 |0.84	| color, **fc** |c(6,5,relu), c(16, 5, relu), c(16, 5, relu), F, d(256, relu) |
|4 | 0.92	| color, fc |c(6,5,relu), c(16, 5, relu), **F, d(256, relu, leakyRelu, 0.5),  d(84, sigmoid)** |
|5 |0.94   | color, **n128** | c(6,5,relu), c(16, 5, relu), F, d(256, relu, leakyRelu, 0.5),  d(84, sigmoid) |
|6 |0.956  after 27 epochs | color, n128| c(6,5,relu, **v**), c(16, 5, relu, **v**), F, d(**128**, leakyRelu, 0.5),  d(84, sigmoid) |
|7 |0.958  after 29 epochs | color, **fc**| c(6,5,relu, v), c(16, 5, relu, v), F, d(128, leakyRelu, 0.5),  d(84, sigmoid) |
Note: **bolded** parameters are the ones that changed from previous try.</br>


Conclution: </br>
* Data preprocessing to keep all channels provides more information to learn than gray scale, as shown in experiment 0 and 1. </br>
* Normalize input image to center close to zero helps training.</br>
    * Normalize by subtractin each image by 128 achieves similar result as featurewise_center (experiment 4,5) </br>
    * Normalizing using featurewise_center will set input mean to 0 over the dataset, this gives higher accuracy
        than set input mean to 0 for each sample image. Since relative intensity difference could be useful in 
        distinguishing one traffic sign from another. Using samplewise_center removed this information and lead
        to lower accuracy. This explains experiment 2 and 3. 
* Padding </br>
    * Same Padding: Has zero padding on boarder to ensure output feature map width and height is the **same** as input width and height. 
        Zero padding allows filter window to slip outside input feature map. 
    * Valid Padding: No padding, filter stays at **valid** positions inside the input feature map. 
        Thus output feature map shrinks by filter_size - 1.

Experiment 7 is the final parameter used. Featurewise centered is used here instead of subtracting 128, since
 it's easier to work with datagenerator. </br>
Keras data generator was used for easier augmentation on images. </br>
```self.train_datagen =ImageDataGenerator(
                            data_format='channels_last',
                            samplewise_center = True,
                            rotation_range=15,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            horizontal_flip=True)
```
                           
Following are explaintion on the shorthand notations I used in the table above:</br>
Validation Accuracy: If epochs is not written, it's the accuracy on the validation set after 30 epochs </br>
Image augmentation notation:</br>
r: rotation_range </br>
w: width_shift_range </br>
h: height_shift_range </br>
z: zoom_range </br>
hf: apply horizontal_flip  </br>
fc: featurewise_center: Boolean. Set input mean to 0 over the dataset, feature-wise.
sc: samplewise_center: Boolean. Set each sample mean to 0. </br>
n128: not use data generator from keras, use `(img-128)/128`
Network architecture notation: </br>
c(k, n, a)  :    conv layer, k filters of size nxn, activation a, v: valild padding</br>
F                     :    flatten </br>
d(n, a, d)               :    dense layer with n nodes, activation a  </br>

Things not changed during parameter tuning: </br>
*Each convolution layer is followed by with max pooling 2x2 and stride 2x2.</br>

*During parameter tuning, final output layer is not change.</br>
Final output layer is a fully connected layer with number of classes nodes, 
and softmax activation to output probability distribution.

## Design and Test a Model Architecture
Preprocessing involves rotation, shift, scaling and horizontal_flip. 
This is implemented using `ImageDataGenerator` in Keras, as shown in `data_augment function` in `utils.py`.

LeNet class in `LeNet.py` defines the model architecture for a modified LeNet structure: <br/>

Use LeNet, inputs are 32x32x3 RGB images , gradient descent is done using Adam optimizer,
 <br/>
During traning, batch size is 128 was used, number of epochs was 20 with early stopping.
Thus training stops when validation loss is not improved for certain number of epochs, which is 5 here. 
Only best model is saved, since without save_best_only, model 5 epochs after the best will be saved. <br/>

Training accuracy was 0.9813, and validation accuracy was 0.956 after 20 epochs. <br/>
## Visualization
Images before and after augmentation are generated for training and testing
images to make sure image formats are correct. <br/>

## Test a Model 
### Test on German Traffic Sign testing dataset
Testing accuracy was 0.939. <br/>

### Test on 5 images from internet
Top 5 softmax probabilities for prediction on each image is generated, and the 
class with highest probability was selected to calculate accuracy. <br/>
Higher softmax probability for a class indicates higher confidence of the model on the prediction. 
