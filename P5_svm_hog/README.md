# Vehicle Detection using HOG and SVM
This project detects vehicles in video using HOG features and SVM for classification.

# Steps
Step 1. Extracted Histogram of Oriented Gradients (HOG) features for images of cars and no-cars.
Step 2. Train a linear support vector machines (SVM) using the HOG features
Step 3. For each frame in the test video, extract HOG feature of the entire frame
Step 4. Use a sliding window to pass reginal HOG feature to SVM for detection 
Step 5. Remove some false positives by constructing and thresholding the heatmap over a history of frames 

# File Structure

```
project
│   writeup_report.md
│   Default Mac desktop Universal          Simulator
└───code
│   |   detection_pipeline.py    Contains main function, input video, output video with bounding box
│   |   feature_util.py          Contains functions for extracting HOG features
|   |   img_util.py              Functions to draw bounding box and generate heatmap
|   |   experiment_param.py      Functions to visualize HOG feature for a set of different parameters
│   
└───SubmissionData
│   └───model/       
|       └─── svc_model.p         svm model, predict car or no-car using HOG feature  
|   └───output_images/   
|       |   project_video.mp4           output video with bounding box  
|       |   heatmap_project_video.mp4   bounding box drawn on heatmap corresponding to project_video.mp4
|       |   project_video_nohist.mp4    output video with bounding box without using history frames to remove false positives
|       |   test_video_thresh_1.mp4    output video on the shorter test video, used threshold = 1 for heatmap 
|       └─── hog_params/               output HOG images with different parameters
```

[//]: # (Image References)
[HSV_HOG]: ./SubmissionData/output_images/hog_params/HSV_ori_15_pixPcell_8_cellPblock_2hog_visuallize.jpg
[RGB_HOG]: ./SubmissionData/output_images/hog_params/RGB_ori_15_pixPcell_8_cellPblock_2hog_visuallize.jpg
[YUV_HOG_8]: ./SubmissionData/output_images/hog_params/YUV_ori_15_pixPcell_8_cellPblock_2hog_visuallize.jpg
[YUV_HOG_16]: ./SubmissionData/output_images/hog_params/YUV_ori_15_pixPcell_16_cellPblock_2hog_visuallize.jpg


[heatmap_thresh]: ./SubmissionData/output_images/heatmap_thresh.jpg
[heatmap]: ./SubmissionData/output_images/heatmap_original.jpg
[bbox_scale_1]: ./SubmissionData/output_images/1_bbox_detected.jpg
[bbox_heatmap]: ./SubmissionData/output_images/bbox_heatmap.jpg


# Experiments

### 1. Histogram of Oriented Gradients (HOG)

Following are the steps of how images are preprocessed for HOG feature extraction, the process of 
choosing prameters for HOG, and the SVM training result for HOG features. 

1. Image preprocess:

StandardScaler() was used for normalizing the HOG features to zero mean and unit variance.
Only the training data was used for calculating the scaler, since the model 
will get information about the test set if the test set was also included in this calculation.
```
train_X = (train_X – mean_trainX) / standard_deviation_trainX
test_X = (test_X – mean_trainX) / standard_deviation_trainX
```

After obtaining the scalar, both the training and test sets were normalized using the scaler.  

Relavent code are in `feature_util.py` implemented as  `prep_feature_dataset`.  

2. HOG is performed using `RGB` images, with 15 HOG orientations, 
using 8  HOG pixels per cell, and 2 HOG cells per block.  

HOG block normalization used 'L2-Hys'. This means L2-norm followed by limiting the maximum 
values to 0.2 (Hys stands for hysteresis) and renormalization using L2-norm.  

HOG features are plotted for varies choice of HOG parameters and color spaces. The ones that 
can generate more distinctive HOG images for car and no-car images are chosen.  

Reason for choose YUV:   

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

Relavent code for visualize the images above are in `experiment_param.py` implemented as `hog_param_vis`. <br\>

3. SVM training using HOG features:<br/>
It took 38.85 Seconds to train the SVM model. Images are normalized, shuffle and split into training and testing
with a ration 8:2. <br/>
Test Accuracy of the model was 0.9703.

Relavent code for visualize the images above are in `detection_pipeline.py` implemented as `train_svm`. <br\>

### 2. Sliding window search

A sliding window approach was used for detection.

Step 1. Extract HOG feature for the entire frame of different scale
For example, when scale = 2, HOG feature is computed on the image downscaled by a factor of 2. 

Step 2. Use sliding window of different scales and different region of interests.<br/>
The same size of sliding window can be used, since the HOG feature is taken at differnent scale,
it's effectively taken HOG feature within different size of windows.

Effects of different parameters for window scale and region of interests location can be judged
by how well the bounding box is covering the vehicles, number of false positives and false negatives.

Step 3. Heatmap was used for further refinement of result.<br/>
Each pixel in the heatmap represent how many times this pixel was inside a bounding box
predicted in Step 2 using different sliding window scales. If a pixel was predicted as vehicle pixel
for multiple bounding box at the same scale, or for bounding boxes at different scales, then
this pixel has a high confidence of being a vehicle. 

Follwoing is a visualization of the sliding window approach. 
![alt text][bbox_scale_1]
Figure 5. Bounding box detected on image without scalinlg. 
![alt text][heatmap]
Figure 6. Heatmap before threshold.
![alt text][heatmap_thresh]
Figure 7. Heatmap after threshold.
![alt text][bbox_heatmap]
Figure 8. Bounding box drawn using thresholded heatmap.



### 3. Video Implementation
Video output has been generated by drawing bounding box on each frame. 

Video without using history frames have a lot of false positives in it: <br/>
https://youtu.be/oztjDQn9O8U
Its corresponding heatmap is shown here: <br/>
https://youtu.be/r0RF7h9dGQI
Relavent implementations are in : `detect_image` function in the `detection_pipeline.py` file. <br/>

Video after using history frames to remove some false positives:<br/>
https://youtu.be/r0V4vmN83nw

### 4. Discussion
A more robust way of detecting vehicles would be use neural networks such as RetinaNet, YOLO, or FasterRCNN.<br/>
Since a lot of parameters for HOG would not work as well when the vehicle size changes. Also, current implementation
has false positives on objects such as trees on the side of road that would generate similar HOG feature
as a vehicle. <br/>
RetinaNet and YOLO are one stage object detectors. Faster RCNN is a two stage object detector.
One stage object detectors usually have lower accuracy than two stage detectors. 
RetinaNet is the first one stage object detector that outperformed two stage object detectors 
in terms of accuracy. 


# References
### Functions from Udacity CarND online course
I built my code based on Udacity CarND online course lesson 20 Object Detection. 
 
# Online githubs
I have taken some ideas from the following githubs, and also the Udacity CarND course.
1. Took idea from the following github to do sliding window on different scales, with different 
region of interests. 
`https://github.com/tatsuyah/vehicle-detection/blob/ed29606d3a66e1ff6d8a4d992f56b52c7a5b1c06/README.md`
I tried to experiment the paramers for scales, ystarts, ystops for better results, but this github's parameters 
always performed better, so I used these parameters in the final result.

2. Used idea from the following github to construct heatmap using historial frames:
`1 + number of bounding box in the most recent 15 frames//2`  
Looking at previous 5 frames, if more than half of the previous frames contain the car, 
then it's a car

`https://github.com/jeremy-shannon/CarND-Vehicle-Detection`

3. Took idea from the following github of not use color hist not so useful, and only used HOG features.
`https://github.com/TusharChugh/Vehicle-Detection-HOG`

