# Advanced Lane Finding Project

---
# Overview 

Goal of the project is to write a software pipeline to identify the lane
boundaries in a video from a front-facing camera on a car. 

Following are the major steps:

* S1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* S2: Apply a distortion correction to raw images.
* S3: Use color transforms, gradients, etc., to create a thresholded binary image.
* S4: Apply a perspective transform to rectify binary image ("birds-eye view").
* S5: Detect lane pixels and fit to find the lane boundary as f(y)
* S6: Determine the curvature of the lane and vehicle position with respect to center.
* S7: Warp the detected lane boundaries back onto the original image, output visual display of the lane boundaries 
       and numerical estimation of lane curvature and vehicle position.


# File Structure

```
project
│   final_writeup.md
└───code
│   |   cameraCalib.py                  Camera calibration 
|   |   img_process.py                  Preprocess image 
│   |   lane_boundary.py                Find lane boundary use sliding window and calculate curvature 
│   |   perspective_transform.py        Warp image 
│   |   pipeline.py                     Main pipeline function
└───data 
|   └───writeup_images/                 Selected output images to use in writeup 
|   └───pipeline_out/                      *.jpg images for visualization
|   └───camera_calib_param/             Save the camera calibration result for later use

```

[//]: # (Image References)
[pinholeCamera]: ./data/writeup_images/pinhole_camera.png
[matrixDecomposition]: ./data/writeup_images/matrix_decomposition.png
[intrinsic_matrix]: ./data/writeup_images/intrinsic_matrix.png
[calibration]: ./data/writeup_images/calibration.png
[chessboard2]: ./data/writeup_images/2.jpg
[chessboard2_undistort]: ./data/writeup_images/2_undist.jpg
[car2]: ./data/writeup_images/test2.jpg
[car2_undistort]: ./data/writeup_images/test2_undist.jpg
[binary_threshold_channels]: ./data/writeup_images/first_frame_channels.jpg
[sobel_hls]: ./data/writeup_images/sobel_hls.png
[closing]: ./data/writeup_images/closing.png
[front_view]: ./data/writeup_images/lines_front.jpg
[bird_eye_view]: ./data/writeup_images/lines_birdeye.jpg
[histogram]: ./data/writeup_images/histogram.jpg
[sliding_window]: ./data/writeup_images/slidingwindow.jpg
[blend_lane]: ./data/writeup_images/blend.jpg
[curvature]: ./data/writeup_images/curvature.png
[line_fit_formula]:./data/writeup_images/line_fit_formula.jpg
# Basic Concepts
####Camera intrinsic and extrinsic matrices
In pinhole camera model, a scene view is formed by projecting 3D points into the image plane 
using a perspective transformation. Following image shows how the variables are defined 
to describe how to transform points in 3D world coordinates to 3D camera coordinates, and how to transform 
points in 3D camera coordinates to 2D homogeneous image coordinates. 

![alt text][pinholeCamera]


Image modified from `https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html`</br>

* `(X, Y, Z)` : coordinates of a 3D point in the world coordinate space</br>
* `(u, v)` : coordinates of the projection point in pixels</br>
* Focal length f: the distance between the pinhole and the film (a.k.a. image plane). `fx, fy` are the focal lengths expressed in pixel units.</br>
* Principal axis: the line perpendicular to the image plane that passes through the pinhole </br>
* Principal point `(cx, cy)`: the intersection of principal axis with the image plane </br>
* Pricipal point offset `x0, y0`: The location of the principal point relative to the film's origin  </br>

Camera intrinsic: 
The focal length and principal offset shows the simple translation of the film relative to the pinhole.</br>
The intrinsic matrix can be seen as a sequence of 2D affine transformations (shear, scaling, and translation transformations) 
that transform 2D homogeneous coordinates [X Y 1] to a new set of 2D points [X' Y' 1]. 

Camera extrinsic parameters: 
[R|t] translates coordinates of a point [X Y Z] to 
a coordinate system. `R` represents rotation matrix, `t` is translation vector.
The camera's extrinsic matrix describes the camera's location in the world, and what direction it's pointing

The intrinsic and extrinsic matrices are summarized below.
![alt text][matrixDecomposition]
Image taken from `http://ksimek.github.io/2013/08/13/intrinsic/`</br>

# Pipeline
#### S1: Camera Calibration 
Implementation is in cameraCalib.py.
Compute the camera calibration matrix and distortion coefficients 

###### S1.1 Detect corners 
`find_corners` function goes through all checkboard images, 
convert to grayscale and use `cv2.findChessboardCorners` to detect corners and
plot them on chessboard images. Next save detected 2d corners in image plane 
as a list `imgpoints`. Also prepare object points, 
like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) and store the 3d points in 3d world coordinate
as a list `objpoints`. </br>

###### S1.2 Calculate intrinsic and extrinsics
Implemented in `pipeline.py` function `calc_intrinsics_test`. 
Opencv function `calibrateCamera` finds the camera intrinsic and extrinsic parameters using `imgpoints` and `objpoints`. 
`cv2.calibrateCamera(objectPoints, imagePoints, imageSize) → retval, cameraMatrix, distCoeffs, rvecs, tvecs`

![alt text][calibration]



As shown in the image above, cv2.calibrateCamera takes as input `m'` and `M'` to calculate intrinsics `A` and 
extrinsics `[R |t]` that satisfies `m' = A[R|t] M'`. 

*Explaination on inputs*

`objectPoints (M')` : 3d points in 3d world coordinate (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0) stored as a list

`imagePoints (m')` :  Detected 2d corners in image plane 

`imageSize`:     (width, height) in pixels

*Explaination on outputs*

`cameraMatrix (A)`: 3x3 intrisic matrix, example output from calibrateCamera() function
```
 [[1.15396093e+03 0.00000000e+00 6.69705357e+02]
 [0.00000000e+00 1.14802496e+03 3.85656234e+02]
 [0.00000000e+00 0.00000000e+00 1.00000000e+00]]

```
`distCoeffs`: distortion coefficients  `(k1, k2, p1, p2, k3)`, `ki` encodes radial distortion,  
`pi` encodes raidal distortion. Example output from calc_calibrateCamera() function:

```angular2html
[[-2.41017956e-01 -5.30721173e-02 -1.15810355e-03 -1.28318856e-04 2.67125290e-02]]
```


`rvecs`: rotation vector, shape = (num images, 3, 1), rotation can be represented by a 3x1 vector since it's 3 degree of freedom.
 
`tvecs`: translation vector, shape = (num images, 3, 1)

Each k-th rotation vector together with the corresponding k-th translation vector 
brings the calibration pattern from the model coordinate space (in which object points are specified) 
to the world coordinate space. This is the extrinsics `[R|t]` mentioned above. 
 
#### S2: Apply a distortion correction to raw images.
For each image read from the video, undistortion was applied first.
using `camera_matrix` and `distorsion_coefficient`.
Following is an example of undistorted images.
* Chessboard with corners detected.

![alt text][chessboard2]  

* Chessboard undistorted

![alt text][chessboard2_undistort]


Following is an example of an image taken from vehicle's perspective.

![alt text][car2]
Following is an example of an image taken from vehicle's perspective after correction.
The result can be observed on the flattened curve from front of vehicle and a less distorted traffic sign.
![alt text][car2_undistort]

#### S3: Binary image of lane lines 
Use color transforms, gradients, etc., to create a thresholded binary image.

######S3.1 Gradient thresholding
Sobel filter is a weighted average of x derivatives or y derivatives. Thus, convolving sobel filter 
over an image can extract the x and y edge responses. Here Sobel x was used, since the lane running in
y direction is of interest. 

![alt text][sobel_hls]

######S3.2 Color thresholding
HSL represents color in hue, saturation and lightness. 

Hue is a degree on the color wheel from 0 to 360. 0 is red, 120 is green, 240 is blue.

Saturation is a measurement of colorfulness. As colors gets closer to white, the saturation value is lower
, whereas colors that are the most intense, like a bright primary color (imagine a bright red, blue, or yellow), 
have a high saturation value. Saturation is a percentage value; 0% means a shade of gray and 100% is the full color. 

Lightness is also a percentage; 0% is black, 100% is white.


######S3.3 Combine gradient and color thresholding

Following are the visualization of steps performed to get the binary image, implemented by function `visualize()`
int `imgProcess.py`

![alt text][binary_threshold_channels]

Meaning on the masking in `combine_thresh` in file `img_process.py`: </br>
`(self.l_binary == 1)`: use lightness to separate bright and dark region , only keep bright region  </br>
`(self.s_binary == 1)`: use saturation to separate out separate tree and yellow lane lines from road </br>
`(self.s_binary == 1) & (self.l_binary == 1)`: color different from road and is not caused by lighting difference </br>
`(self.sxbinary == 1)`: use sobel filter to find edges 
`(self.s_binary == 1) & (self.l_binary == 1) | (self.sxbinary == 1)` : 
color different from road and is not caused by lighting difference, or is at edge found by sobel filter. </br>

######S3.4 Remove noise from binary images 
Opencv function `Closing` is the dilation followed by erosion. 
It is useful in closing small holes inside the foreground objects, or small black points on the object.
![alt text][closing]

The thresholded binary image result returned by function `combine_thresh` is used in following steps. 


#### S4: Apply a perspective transform to rectify binary image ("birds-eye view").
In order to warp the front view to a birds-eye view, 4 corresponding points are needed in the original image
and birds-eye view image. The 4 points are hand selected being points on the lanes, since the left and right 
lanes should become parallel after warping, as shown in the following images.

Front view: 

![alt text][front_view]

Birds-eye view:

![alt text][bird_eye_view]

The function `warp_front_to_birdeye` in `perspectiveTransform.py` only need to be called once for the first frame
to calculate the perspective transformation, and the same transformation matrix can be used for
following frames. 

#### S5: Detect lane pixels and fit to find the lane boundary as f(y)

###### S5.1 Locate peak using histogram

After applying calibration, thresholding, and a perspective transform to a road image, 
this step decides the approximate location of left line and right line.

This is achieved by taking a histogram along all the columns in the lower half of the image,
and find the peak of the left and right halves of the histogram. 
These will be the starting point for the left and right lines.

Following is the histogram. The x coordinates that gives the largest count for left and 
right half of the image would correspond to the left and right lanes, which are
284 and 1137 in the x coordinates shown in the following histogram. This is implemented in `histogram_peaks` function
in `laneBoundary.py`. 

![alt text][histogram]

This step is only needed for the first frame, since the later frames only needs to search in a margin around 
the previous line position.


###### S5.2 Detect lane
Lane detection for the first frame uses sliding window as implemnted in `slide_window` in `laneBoundary.py`.
Use the x coordinate for the two highest peaks from the histogram as a starting point
for determining where the lane lines are. Then use sliding windows moving upward in the 
image (further along the road) to determine where the lane lines go.

Use the x coordinate found in previous step for left and right lane as 
a starting point. Then create a window centered at the x coordinate for
each lane, as shown in `create_window` function in `laneBoundary.py`.

Next, find indices for pixels within the current window that have nonzero intensities.
If the count of nonzero pixels is more than a threshold, 
recenter the window using the mean x-coordinate of these nonzero pixels, as shown in the following image.
The green boxes are the search windows, the red and blue highlighted pixels 
are pixels belonging to the lanes after sliding window search. 

![alt text][sliding_window]

If the frame is not the first frame, the search only needs to happend in a margin around the
 previous lane position, as implemented in `fit_use_pref`.
 
In both cases, input was first frame or not, the indices for nonzero pixels that should
belong to left or right lane are stored as a numpy array for the next step. These are the red and blue highlighted pixels 
in the image above. 

###### S5.3 Fit lane
This step fits f(y) rather than f(x) because the lane lines in the warped image
is nearly vertical and may have the same x value for more than one y value.

The lane function f(y) is approximated using a second order polynomial. 
The fitted line is plotted in yellow shown in the image in S5.2.
 
Following is the mathmatical formula illustrated on image from Udacity. 
 
![alt text][line_fit_formula]
#### S6: Determine the curvature of the lane and vehicle position with respect to center.

###### S6.1 Lane Curvature
Radius of curvature intuition:
The radius of curvature of the curve at a particular point is defined as the radius of the approximating circle. 
This radius changes as we move along the curve. The curvature depends on the radius, 
the smaller the radius, the greater the curvature.

The curvature calculation has to be performed on the frame that's warped to the birdeye view.

When calculating the curve from the car view, the curvature obtained is not the "real" curvature 
because it's seen in a different perspective. This is implemented in `calc_curvature` in file `laneBoundary.py`.

The y values of the image increase from top to bottom, thus the measure the radius of curvature closest to vehicle
can be obtained by evaluating the formula above at the y value corresponding to the bottom of image.

Following is the mathmatical explaination of curvature provided by Udacity:

![alt text][curvature]



###### S6.2 Vehicle Position
Assume center of vehicle is at the center of image, and take the center between the left and right lanes
near the bottom of image (close to the vehicle side not the horizon side) as center of road.

Then convert the distance between vehicle center and lane center from pixel to meters. 

U.S. regulations that require a minimum lane width is 3.7 meters, corresponding to 
the left and right lines separation of roughly 700 pixels in x-dimension.
The video data provided that the line is 30 meters long in y-dimension. 

Therefore, to convert from pixels to real-world meter measurements:

```apple js
ym_per_pix = 30/720    // meters per pixel in y dimension
xm_per_pix = 3.7/700  //meters per pixel in x dimension
```

#### S7: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

First, create an image with left and right lanes plotted bird-eye view, then warp it to front view. 
This image with lanes plotted was blended with original front view image, as shown below. 

The part is implemented in `visualize_lane`.

![alt text][blend_lane]


# Reference
* Related concepts and images

`http://ksimek.github.io/2012/08/22/extrinsic/` 
`https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html`</br>
`http://ksimek.github.io/2013/08/13/intrinsic/`
`https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html`
* Sobel filter and dilation images are from CMU computer vision course







