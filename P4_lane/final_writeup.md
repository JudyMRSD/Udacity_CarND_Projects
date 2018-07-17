# Advanced Lane Finding Project

---
# Overview 

Goal of the project is to write a software pipeline to identify the lane
boundaries in a video from a front-facing camera on a car. Following are the 
major steps:

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

# Pipeline
#### S1: Camera Calibration 
Implementation is in cameraCalib.py.
Compute the camera calibration matrix and distortion coefficients 

Given a set of chessboard images, 

TODO:
Briefly state how you computed the camera matrix and distortion coefficients. 
Provide an example of a distortion corrected calibration image.
TODO:
Provide an example of a distortion-corrected image.


#### S2: Apply a distortion correction to raw images.
#### S3: Binary image of lane lines 
Use color transforms, gradients, etc., to create a thresholded binary image.
TODO: Provide an example of a binary image result.
Meaning on the masking in `combine_thresh` in file `img_process.py`: </br>
`(self.l_binary == 1)`: use lightness to separate bright and dark region , only keep bright region  </br>
`(self.s_binary == 1)`: use saturation to separate out separate tree and yellow lane lines from road </br>
`(self.s_binary == 1) & (self.l_binary == 1)`: color different from road and is not caused by lighting difference </br>
`(self.sxbinary == 1)`: use sobel filter to find edges 
`(self.s_binary == 1) & (self.l_binary == 1) | (self.sxbinary == 1)` : 
color different from road and is not caused by lighting difference, or is at edge found by sobel filter. </br>


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























