from cameraCalib import CameraCalibration
from img_process import Image_Process
from perspective_transform import Perspective
import os
import numpy as np
import cv2
import matplotlib.image as mpimg
Data_Dir = '../data/'
Chessboard_In_Dir = Data_Dir+'camera_cal/'
Chessboard_Out_Dir = Data_Dir+'output_images/camera_cal_out/'
Save_Params_Path = Data_Dir+'camera_calib_param/dist_pickle.p'
Chessboard_Input_Test = Data_Dir+'camera_cal/calibration1.jpg'
Chessboard_Output_Test = Data_Dir+'output_images/camera_cal_out/calibration1_undist.jpg'
Car_Input_Test = Data_Dir+'test_images/straight_lines2.jpg'
Car_Output_Test = Data_Dir+'output_images/test_image_out/straight_lines2_out.jpg'
Input_Dist_Pkl = Data_Dir+'camera_calib_param/dist_pickle.p'

outdir = Data_Dir+'output_images/thresh_out/'
input_img_path = Data_Dir+'test_images/test5.jpg'


# Camera calibration params for checker board
Corner_Rows = 6
Corner_Cols = 9


class Pipeline:
    def __init__(self):
        pass

    def check_paths(self):
        # check correctness of file paths
        assert (os.path.exists(Chessboard_In_Dir)), "ERROR: Chessboard_In_Dir does not exist"
        assert (os.path.exists(Chessboard_Out_Dir)), "ERROR: Chessboard_Out_Dir does not exist"
        assert (os.path.exists(Save_Params_Path)), "ERROR: Save_Params_Path does not exist"
        assert (os.path.exists(Chessboard_Output_Test)), "ERROR: Chessboard_Output_Test does not exist"
        assert (os.path.exists(Car_Input_Test)), "ERROR: Car_Input_Test does not exist"

    # detect lane lines in one image
    def detect_img(self):
        calibTool = CameraCalibration()
        # Step 1 : Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
        calibTool.calc_param(Chessboard_In_Dir + "*.jpg", Chessboard_Out_Dir, corner_rows=Corner_Rows,
                             corner_cols=Corner_Cols, save_params_path=Save_Params_Path)
        # calibTool.undistort(chessboard_input_test, chessboard_output_test,input_dist_pkl)


def main():

    pl = Pipeline()
    pl.check_paths()
    pl.detect_img()

    # # Step 2: Apply a distortion correction to raw images.
    # undistort_front = calibTool.undistort(car_input_test, car_output_test, input_dist_pkl)
    #
    # # Step 3: Use color transforms, gradients, etc., to create a thresholded binary image.
    # pipeline = Image_Process()
    # pipeline.visualize(undistort_front)
    #
    # binary_front_img = pipeline.closing
    # print("img shape" , binary_front_img.shape)
    # cv2.imwrite(outdir + "binary_front_img.jpg", binary_front_img)
    #
    #
    # # Step 4: Apply a perspective transform to rectify binary image ("birds-eye view").
    # # 1. undistort use matrix from cameraCalib.py
    # # 2. binarize: only pixels on the lane is bright
    # # 3. warp
    #
    # perspective_tool = Perspective()
    #
    # h, w = binary_front_img.shape[0:2]
    # src = np.float32([[w, h],
    #                   [0, h],
    #                   [550, 460],
    #                   [730, 460]])
    # dst = np.float32([[w, h],
    #                   [0, h],
    #                   [0, 0],
    #                   [w, 0]])
    #
    # binary_front_img = cv2.cvtColor(binary_front_img, cv2.COLOR_GRAY2BGR)
    #
    # birdeye_img, to_bird_matrix, to_front_matrix = perspective_tool.warp_front_to_birdeye(src, dst,
    #                                                                                       binary_front_img,
    #                                                                                       verbose=True)
    # print("to_front_matrix", to_front_matrix)
    #
    # # plot the 4 lines on original image
    # # plot warp the image with lines plotted on it
    # # 4. region of interest
    #
    #
    # # Step 5: Detect lane pixels and fit to find the lane boundary.
    #


if __name__ == "__main__":
    main()