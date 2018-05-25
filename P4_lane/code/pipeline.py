from cameraCalib import CameraCalibration
from img_process import Image_Process
import os

import matplotlib.image as mpimg


def main():


    chessboard_in_paths = '../camera_cal/*.jpg'
    chessboard_out_paths = '../output_images/camera_cal_out/'
    save_params_path = '../output_numeric/dist_pickle.p'
    chessboard_input_test = '../camera_cal/calibration1.jpg'
    chessboard_output_test = '../output_images/camera_cal_out/calibration1_undist.jpg'
    car_input_test = '../test_images/test2.jpg'
    car_output_test = '../output_images/test_image_out/test2_undist.jpg'
    input_dist_pkl = '../output_numeric/dist_pickle.p'

    outdir = '../output_images/thresh_out/'
    input_img_path = '../test_images/test5.jpg'

    corner_rows = 6
    corner_cols = 9

    calibTool = CameraCalibration()
    # Step 1 : Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

    #calibTool.calc_param(chessboard_in_paths, chessboard_out_paths, corner_rows=corner_rows, corner_cols=corner_cols, save_params_path=save_params_path)
    # calibTool.undistort(chessboard_input_test, chessboard_output_test,input_dist_pkl)

    # Step 2: Apply a distortion correction to raw images.
    #calibTool.undistort(car_input_test, car_output_test, input_dist_pkl)

    # Step 3: Use color transforms, gradients, etc., to create a thresholded binary image.
    # image = mpimg.imread(input_img_path)
    # base = os.path.basename(input_img_path)
    # base = os.path.splitext(base)[0]
    #
    # pipeline = Image_Process(outdir, base, image)
    # pipeline.visualize()

    # Step 4: Apply a perspective transform to rectify binary image ("birds-eye view").
    # 1. undistort use matrix from cameraCalib.py
    # 2. binarize: only pixels on the lane is bright
    # 3. warp
    # plot the 4 lines on original image
    # plot warp the image with lines plotted on it
    # 4. region of interest


    # Step 5: Detect lane pixels and fit to find the lane boundary.
    


if __name__ == "__main__":
    main()