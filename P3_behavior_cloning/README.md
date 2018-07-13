# Files
```
project
│   writeup_report.md
│   Default Mac desktop Universal          Simulator
└───code
│   |   model.py    create and train model
│   |   util.py     helper functions
|   |   drive.py    Udacity's script to drive the car
│   
└───data
│   |   driving_log.csv    stores file paths for training images and corresponding steering angle
│   └───model/       
|       └─── model.h5 
|   └───result_video/  *.MOV file recording driving for one track 
|   └───writeup_imgs/   images saved for writeup 
|   └───debug/   images saved for debugging
|   └───envs/    anaconda environment files *.yml format 
```

[//]: # (Image References)
[elu]: ./data/writeup_imgs/elu.png
[angle_original]: ./data/writeup_imgs/original_train_angle.png
[angle_flip]: ./data/writeup_imgs/after_flip_train_angle.png
[angle_filter]: ./data/writeup_imgs/remove_small_angles.png
[vis_aug_angle_turn]: ./data/writeup_imgs/vis_aug_imgs_turn.jpg
[vis_aug_angle_straight]: ./data/writeup_imgs/vis_aug_imgs.jpg


# Data
driving_log.csv example data format: 

|center	| left	|  right  |  steering | throttle  |	brake  |   speed  |
|-------|-------|---------|-----------|-----------|--------|----------|
|IMG/center_2016_12_01_13_30_48_287.jpg	| IMG/left_2016_12_01_13_30_48_287.jpg| IMG/right_2016_12_01_13_30_48_287.jpg|0|0|0	|22.14829|


# environment
`conda env create -f keras_carnd_p3_gpu_works.yml`       
If tensorflow is shown as anaconda package, uninstall and reintall use pip: 
(precompiled tensorflow 1.5 works with CUDA versions <= 9.0)
`pip install tensorflow==1.4`
`pip install tensorflow-gpu==1.4`

run on Ubuntu with GPU:   `source activate keras_carnd_p3`

# Implementation details
#### Generator
A python generator was used in `model.py` to generate data for trianing 
rather than storing the trainign data in memory. <\br>
Implementation of a generator with data augmentation is function `generator` in `util.py`. 

How generator works in python: (cited from Udacity course)
"
Generators can be a great way to work with large amounts of data. Instead of storing the preprocessed data 
in memory all at once, using a generator you can pull pieces of the data and process them on the fly only 
when you need them, which is much more memory-efficient.

A generator is like a coroutine, a process that can run separately from another main routine, which makes 
it a useful Python function. Instead of using return, the generator uses yield, which still returns the 
desired output values but saves the current values of all the generator's variables. 
When the generator is called a second time it re-starts right after the yield statement, 
with all its variables set to the same values as before.
"

#### Image format
Training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.
Since the input for `drive.py` need to match the input image shape used for model training, a preprocessing
step was added to downsample image to half the width and height. 

#### Training
Model was trained for 20 epochs, `samples_per_epoch = 32000` means 3200 images were used for
 training during one epoch. Early stopping along with saving the best model
 according to the validation loss.
 

# Model 

#### Activation function choice
Activation function was used for introducing nonlinearity to the network.
Non-linear means that the output (i.e. steering angles) cannot be reproduced 
from a linear combination of the inputs (i.e. images). 

RELU would have "dying RELU" problem, which means the RELU slop is always zero 
when for negative input, thus the gradient `dW/dx` for backpropagation will become
zero.

ELU is similar to RELU but solves the "dying RELU" issue. 
Both ELU and RELU are identity function for non-negative inputs. 
ELU becomes smooth slowly until its output equal to `-a` where RELU sharply becomes 0. 
![alt text][elu]

#### Loss function and gradient descent
Mean squared loss was used as loss function. Gradient descent was 
performed using Adam optimizer.

#### Model architecture
I'm using the simple model architecture from the following blog post:
`https://medium.com/@ValipourMojtaba/my-approach-for-project-3-2545578a9319`
The first two layers are preprocessing layers. Layer `cropping2d_1 ` cropped image
keep region of interst and remove trees and sky. Layer `lambda_1` normalized input 
using `(x / 255.0) - 0.5` to have zero mean. Since all features should be normalized
so that each feature have same amount of influence on the loss calculation 
for backpropagation. Otherwise gradient descent would suffer from oscilation and hard
to converge. 

The model used 3 convolution layers with exponential linear unit `elu` activation, 
maxpooling and 0.3 dropout ratio. The conv layer feature map was then flattened
and passed through 4 fully connected layers with 0.5 dropout. The final layer outputs a single
value representing steering angle, a value between -1 (sharp left turn) 
and 1 (sharp right turn). 

Dropout is used for reduce overfitting. With dropout, at each hidden layer,
the neurons are will not participate in forward and backward propagation with
a probability of p. Thus, the network learns a more generatl mapping between
 input and output.

image reference: http://prog3.com/sbdm/blog/cyh_24/article/details/50593400

The architecture is shown below: </br>
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
cropping2d_1 (Cropping2D)    (None, 40, 160, 3)        0         
_________________________________________________________________
lambda_1 (Lambda)            (None, 40, 160, 3)        0         
_________________________________________________________________
conv_1 (Conv2D)              (None, 40, 160, 3)        12        
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 33, 153, 16)       3088      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 16, 76, 16)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 16, 76, 16)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 12, 72, 32)        12832     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 6, 36, 32)         0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 6, 36, 32)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 4, 34, 32)         9248      
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, 17, 32)         0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 2, 17, 32)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 1088)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 256)               278784    
_________________________________________________________________
dropout_4 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 128)               32896     
_________________________________________________________________
dropout_5 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_3 (Dense)              (None, 64)                8256      
_________________________________________________________________
dropout_6 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 8)                 520       
_________________________________________________________________
dropout_7 (Dropout)          (None, 8)                 0         
_________________________________________________________________
output_angle (Dense)         (None, 1)                 9         
=================================================================
Total params: 345,645
Trainable params: 345,645
Non-trainable params: 0
_________________________________________________________________

```

# Data Augmentation

Training data was preprocessed to ensure a balanced dataset that is not overwhelmed by 
zero angles or left turn agnles. Testing data, which are images from simulation during
testing, are not augmented. Following are the detailed steps tried. 

Original angle distribution, left and right turn are not balanced. 
![alt text][angle_original]

Angle distribution using 100 original images after flipping to balance left and right turns.
This also doubled up the amount of data: 
![alt text][angle_flip]

Angle distribution after flip, light adjustment, and shifts are plotted in the image below.
To balance zero and nonzero angles, 25% of angles close to zero are kept. </br>
For each set of center, left and right images, adjust center angle for left and right image. </br>
Each of the 3 images are flipped to balance left and right turns. </br>
Shift is applied to simulate cars at different locations in the lane. </br>
    
The angles now are much more uniformly distributed. Light adjustment was introduced by 
adding random noise to v channel in hsv format image. This light augmentation was not used in final model
training, since it didn't give improvement on result. 
![alt text][angle_filter]

Following are two images showing how images are adjusted during data augmentation. 
The green line indicates the steering angle after adjustment. As shown in the images,
 the adjusted angle kept the end point of the green line at center of frame after data
 augmentation. Angles are between -1 and 1. 
![alt text][vis_aug_angle_turn]
![alt text][vis_aug_angle_straight]

# Lessons learned for parameter tuning
Input image changed, the same model architecture might not work.
For example, if image A is twice the width and height of image B,
the feature map obtained after passing both images through the same conv layers
will be different. Image A will produce a larger feature map, thus need
an architecture that can learn more complex function.

# References
I was using Udacity's CarND tutorial code as a starting baseline, but the car 
didn't learn how to drive. 

Method to preprocess data and build model architecture:
The data augmentation method for shifting image and parameters for it were taken from: 
`https://medium.com/@ValipourMojtaba/my-approach-for-project-3-2545578a9319`
Shift in y direction up to 10 pixels, applied angle change of .2 per pixel.
(jin: ~60 degrees very sharp turn if the vehicle is close to the edge of road (center of road is at edge of camera frame))
https://github.com/vxy10/P3-BehaviorCloning

# Result 
The vehicle was able to successfully complete both tracks. </br>
Track 1: The model was trained using images from track 1. </br>
`https://youtu.be/movG40dYOUY`
Track 2: The model has never seen track 2 images during triaing but still 
successfully traversed it. </br>
`https://youtu.be/E3roWjnz_Ts`
