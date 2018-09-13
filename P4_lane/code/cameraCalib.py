# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

# code is modified based on examples from Udacity oneline course
# https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb
import pickle
import numpy as np
import cv2
import glob
import os
Debug = True
class CameraCalibration:
    def __init__(self):
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane.
    @staticmethod
    def getbase(filename):
        # input: filename      integer.suffix
        base = os.path.basename(filename)
        file_index = int(os.path.splitext(base)[0])
        return file_index

    # corner_rows  # number of corners (not count ones on chessboard boarder
    def find_corners(self, chessboard_in_paths, chessboard_out_paths, corner_rows, corner_cols):
        image_paths = sorted(glob.glob(chessboard_in_paths), key=self.getbase)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((corner_cols* corner_rows,3), np.float32)
        objp[:,:2] = np.mgrid[0:corner_cols, 0:corner_rows].T.reshape(-1,2)
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(image_paths):
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (corner_cols, corner_rows), None)
            # If found, add object points, image points
            if ret == True:
                print("corners detected fname", fname)
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (corner_rows,corner_cols), corners, ret)
                write_name = chessboard_out_paths+str(idx+1)+'.jpg'
                cv2.imwrite(write_name, img)
            else:
                print("specified amount of corners not detected from fname", fname)
        self.img_size = (img.shape[1], img.shape[0])
        print("img size", self.img_size)

    def calc_param(self,chessboard_in_paths, chessboard_out_paths, corner_rows, corner_cols, save_params_path):
        print("find corners")
        self.find_corners(chessboard_in_paths, chessboard_out_paths, corner_rows, corner_cols)
        # Do camera calibration given object points and image points
        # Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.

        # self.camera_matrix  3x3     [[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]]
        # self.dist_coeff     distortion coefficient    [k1, k2, p1, p2 [k3, k4, k5, k6]]
        # rvecs, tvecs are rotation and translation vectors representing the transformation from 3D world to 3d camera coordinate

        ret, self.camera_matrix, self.dist_coeff, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, None, None)
        if Debug:
            print("Camera matrix (intrinsic matrix) ", self.camera_matrix.shape)
            print("distortion coefficients", self.dist_coeff)
            print("rotation vectors shape ", np.array(rvecs).shape)
            print("translation vectors shape", np.array(tvecs).shape)

            print("rotation vectors ", rvecs)
            print("translation vectors", tvecs)
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["camera_matrix"] = self.camera_matrix
        dist_pickle["distorsion_coefficient"] = self.dist_coeff
        pickle.dump(dist_pickle, open(save_params_path, "wb"))

    # # undistort and save result if output_test path is valid
    # def undistort(self, input_test, input_dist_pkl, output_test= None):
    #     dist_pickle = pickle.load(open(input_dist_pkl, 'rb'))
    #     camera_matrix = dist_pickle["camera_matrix"]
    #     distorsion_coefficient = dist_pickle["distorsion_coefficient"]
    #     print("camera_matrix", camera_matrix)
    #     print("distorsion_coefficient", distorsion_coefficient)
    #     # Test undistortion on an image
    #     img = cv2.imread(input_test)
    #     # undistort: Transforms an image to compensate for lens distortion.
    #     dst = cv2.undistort(img, camera_matrix, distorsion_coefficient, None, camera_matrix)
    #     if(output_test):
    #         cv2.imwrite(output_test,dst)
    #     return dst


# main function for unit test of cameraCalib.py
def main():

    data_dir = "../data/"
    assert (os.path.exists(data_dir)), "ERROR: data_dir does not exist"
    chessboard_in_paths = data_dir+'camera_cal/*.jpg'
    chessboard_out_paths = data_dir+'output_images/camera_cal_out/'
    save_params_path = data_dir+'camera_calib_param/dist_pickle.p'
    chessboard_input_test = data_dir+'camera_cal/2.jpg'
    chessboard_output_test = data_dir+'output_images/undistort/calibration2_undist.jpg'

    car_input_test = data_dir+'test_images/test2.jpg'
    car_output_test = data_dir+'output_images/undistort/calibration2_undistort.jpg'
    input_dist_pkl = data_dir+'camera_calib_param/dist_pickle.p'

    corner_rows = 6
    corner_cols = 9

    calibTool = CameraCalibration()
    calibTool.calc_param(chessboard_in_paths, chessboard_out_paths, corner_rows=corner_rows, corner_cols=corner_cols, save_params_path=save_params_path)
    # calibTool.undistort(chessboard_input_test,input_dist_pkl, chessboard_output_test)
    # calibTool.undistort(car_input_test, input_dist_pkl, car_output_test)

if __name__ == "__main__":
    main()