# TODO:
remove main function (unit test) from the helper files

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
* S7: Warp the detected lane boundaries back onto the original image.
* S8: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


# File Structure

```
project
│   final_writeup.md
└───code
│   |   cameraCalib.py                  Camera calibration 
│   |   colorSpace.py                   Visualize color space  
|   |   img_process.py                  Preprocess image 
│   |   lane_boundary.py                Find lane boundary use sliding window and calculate curvature 
│   |   perspective_transform.py        Warp image 
│   |   pipeline.py                     Main pipeline function
└───data 
|   └───output_images/                
|   └───writeup_images/                 Selected output images to use in writeup 
|   └───visualize/                      *.jpg images for visualization
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

<table style="width:100%">
  <tr>
    <th>
      <p align="center">
           <img src="./img/./data/writeup_images/2.jpg" alt="sliding_windows_before" width="60%" height="60%">
           <br> Original checkerboard
      </p>
    </th>
    <th>
      <p align="center">
           <img src="./img/./data/writeup_images/2_undist.jpg" alt="sliding_windows_after" width="60%" height="60%">
           <br>After undistortion
      </p>
    </th>
  </tr>
</table>



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



#### S5: Detect lane pixels and fit to find the lane boundary as f(y)

Fitting f(y) rather than f(x) because the lane lines in the warped image
is nearly vertical and may have the same x value for more than one y value.

#### S6: Determine the curvature of the lane and vehicle position with respect to center.

Radius of curvature intuition:
The radius of curvature of the curve at a particular point is defined as the radius of the approximating circle. 
This radius changes as we move along the curve. 

The curvature calculation has to be performed on the frame that's warped to the birdeye view.

When calculating the curve from the car view, the curvature obtained is not the "real" curvature 
because it's seen in a different perspective. 

When we calculate throw the bird-eye view, we have the true curvature of the lane related to the ground.

#### S7: Warp the detected lane boundaries back onto the original image.
#### S8: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
 
Assumptions:

1. lane is about 30 meters long and 3.7 meters wide.</br>
2. center of image is along the center line of vehicle </br>













# Reference
* Related concepts and images
`http://ksimek.github.io/2012/08/22/extrinsic/`</br>
`https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html`</br>
`http://ksimek.github.io/2013/08/13/intrinsic/`
`https://docs.opencv.org/3.4/d9/d61/tutorial_py_morphological_ops.html`
* Sobel filter image
CMU computer vision course







