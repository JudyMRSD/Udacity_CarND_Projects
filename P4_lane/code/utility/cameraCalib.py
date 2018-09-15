# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
import pickle
import numpy as np
import cv2
import glob
import os

class CameraCalibration:
    def __init__(self):
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d points in world coordinates
        self.imgpoints = []  # 2d points in image plane

    def find_corners(self, chessboard_in_dir, chessboard_out_dir, corner_rows, corner_cols):
        '''
        Detect chessboard corners and visualize
        :param chessboard_in_dir: Directory containing chessboard images
        :param chessboard_out_dir: Directory to store chessboard images with detected corners
        :param corner_rows: Ground truth number of rows for chessboards
        :param corner_cols: Ground truth number of columns for chessboards
        '''
        image_paths = glob.glob(chessboard_in_dir+"*.jpg")
        print("image_paths", image_paths)
        # prepare object points in 3d world coordinate, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((corner_cols* corner_rows,3), np.float32)
        objp[:,:2] = np.mgrid[0:corner_cols, 0:corner_rows].T.reshape(-1,2)
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(image_paths):
            print("fname", fname)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (corner_cols, corner_rows), None)
            # If found, add object points, image points
            if ret == True:
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (corner_rows,corner_cols), corners, ret)
                write_name = chessboard_out_dir + str(idx + 1) + '.jpg'
                cv2.imwrite(write_name, img)
            else:
                print("specified amount of corners not detected from fname", fname)
        self.img_size = (img.shape[1], img.shape[0])

    def calc_param(self, chessboard_in_dir, chessboard_out_dir, corner_rows, corner_cols, save_params_path):
        '''
        Do camera calibration given object points and image points
        Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
        self.camera_matrix  3x3     [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]
        self.dist_coeff     distortion coefficient    [k1, k2, p1, p2 [k3, k4, k5, k6]]
        rvecs, tvecs are rotation and translation vectors representing the transformation from 3D world to 3d camera coordinate
        '''
        self.find_corners(chessboard_in_dir, chessboard_out_dir, corner_rows, corner_cols)
        ret, self.camera_matrix, self.dist_coeff, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, None, None)

        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["camera_matrix"] = self.camera_matrix
        dist_pickle["distorsion_coefficient"] = self.dist_coeff
        pickle.dump(dist_pickle, open(save_params_path, "wb"))
