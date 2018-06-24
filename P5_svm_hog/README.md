# File Structure
output_images:   all results saved for debugging
writeup_results: results for writeup


[//]: # (Image References)
[HSV_HOG]: ./SubmissionData/output_images/hog_params/HSV_ori_15_pixPcell_8_cellPblock_2hog_visuallize.jpg
[RGB_HOG]: ./SubmissionData/output_images/hog_params/RGB_ori_15_pixPcell_8_cellPblock_2hog_visuallize.jpg
[YUV_HOG_8]: ./SubmissionData/output_images/hog_params/YUV_ori_15_pixPcell_8_cellPblock_2hog_visuallize.jpg
[YUV_HOG_16]: ./SubmissionData/output_images/hog_params/YUV_ori_15_pixPcell_16_cellPblock_2hog_visuallize.jpg
[Heat_map]: TODO: add heatmap image 
[video1]: ./SubmissionData/project_video.mp4



# Experiments

### Parameters:
HOG is performed using `RGB` images, with 15 HOG orientations, 
using 8  HOG pixels per cell, and 2 HOG cells per block. <br/>
HOG block normalization used 'L2-Hys'. This means L2-norm followed by limiting the maximum 
values to 0.2 (Hys stands for hysteresis) and renormalization using L2-norm. 

Heat map with pixel values larger than 1 are treated as detected vehicle pixels. <br/>

It took 38.85 Seconds to train SVC, with 80% images as training, and 20% as validation. 
Test Accuracy of SVC =  0.9703



### Reason for Choose parameters:

HOG features are plotted for varies choice of HOG parameters and color spaces. The ones that 
can generate more distinctive HOG images for car and no-car images are chosen.

Reason for choose YUV:  <br/>
YUV gives more distinct HOG features between car and no-car images across all channels, as shown
 in Figure 1, 2, 3, which are HOG features for each channel in HSV, RGB and YUV color spaces.<br/>
The first row holds car images, the second row is for no-car images.<br/>

Reason for choose 8 pixels per cell: <br/>
8 pixels per cell gives richer details. As shown in Figure 3 and Figure 4, YUV using 8 pixels per cell
 has more details in HOG image than using 16 pixels per cell.<br/>

![alt text][HSV_HOG]
Figure 1: HSV 15 orientations, 8 pixels per cell, and 2 cells per block:<br/><br/><br/>

![alt text][RGB_HOG]
Figure 2: RGB 15 orientations, 8 pixels per cell, and 2 cells per block:<br/><br/><br/>

![alt text][YUV_HOG_8]
Figure 3: YUV 15 orientations, 8 pixels per cell, and 2 cells per block:<br/><br/><br/>

![alt text][YUV_HOG_16]
Figure 4: YUV 15 orientations, 16 pixels per cell, and 2 cells per block:<br/><br/><br/>




Scp source dest 

Upload detection_refactor/

scp -r ./detection_refactor/ jinz1@128.237.99.172:/home/jinz1/Jin/Intersection_TrafficFlow/

Download snapshots 
scp -r jinz1@128.237.99.172:/home/jinz1/Jin/Intersection_TrafficFlow/detection/detector/snapshots ./






# TODO
1. make the svm train parameter same as testing part
2. feature extraction combine code for individual window and whole image

# functions from Udacity CarND online course
draw_boxes
inside Course_code/6_svm_hog/Code

train_svm  is from search_classify.py
find_cars   from    hog_subsample.py

# Online githubs
I have taken some ideas from the following githubs, and also the Udacity CarND course.
1. Took idea from the following github to do sliding window on different scales, with different 
region of interests. 
`https://github.com/tatsuyah/vehicle-detection/blob/ed29606d3a66e1ff6d8a4d992f56b52c7a5b1c06/README.md`

2. Used idea from the following github to construct heatmap using historial frames:
1 + len(det.prev_rects)//2 
Looking at previous 15 frames, if more than half of the previous frames contain the car, 
then it's a car

`https://github.com/jeremy-shannon/CarND-Vehicle-Detection`

3. Took idea from the following github of not use color hist not so useful, and only used HOG features.
`https://github.com/TusharChugh/Vehicle-Detection-HOG`

# example save trained model

# accuracy
Images are shuffled and spitted into train , test
RGB --> YUV --> HOG

0.9     use 50 images no car and car 
0.9623  use all images 

0.15 test data

Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 5292
32.02 Seconds to train SVC...
Test Accuracy of SVC =  0.9595

# result 
55.5 Seconds to train SVC...
Test Accuracy of SVC =  0.9688


HOG hyper parameter:
block_norm = 'L2-Hys'
Normalization using L2-norm, followed by limiting the maximum values to 0.2 
(Hys stands for hysteresis) and renormalization using L2-norm

# Explain Function implementations

input:
HOG parameters,
car images folder (N imgs),
no-car images folder (M imgs)
flag visualize : save HOG feature and HOG hist
output:
train set, test set
txt file, N rows, each rows stores [HOG feature flatten and normalized   |  label]
(label = 1 for vehicle, label = 0 for non-vehicle)

steps:
read images from vehicle directory,and non-vehicle directory

extract HOG feature for each image
car :  n x [HOG_featuer_length | 1 (class)] numpy array
no car: m x [HOG_featuer_length | 0 (class)] numpy array

flag visualize = true:   save one car and no-car HOG feature as image
and histogram image

shuffle and split into test and train set   8:2
test: num_test x [HOG_featuer_length | class] numpy array
train: num_train x [HOG_featuer_length | class] numpy array

