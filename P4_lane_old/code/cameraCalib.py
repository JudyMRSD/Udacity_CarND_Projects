# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

# code adapted from Udacity oneline course
# https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb


import pickle
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


class CameraCalibration:
    def __init__(self):
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d points in real world space
        self.imgpoints = []  # 2d points in image plane.

    # corner_rows  # number of corners (not count ones on chessboard boarder
    def find_corners(self,chessboard_in_paths, chessboard_out_paths, corner_rows, corner_cols):
        image_paths = glob.glob(chessboard_in_paths)
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((corner_cols* corner_rows,3), np.float32)
        objp[:,:2] = np.mgrid[0:corner_cols, 0:corner_rows].T.reshape(-1,2)
        print("image_paths", image_paths)
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(image_paths):
            print("idx", idx)
            img = cv2.imread(fname)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (corner_cols, corner_rows), None)

            # If found, add object points, image points
            if ret == True:
                print("corners detected fname", fname)
                self.objpoints.append(objp)
                self.imgpoints.append(corners)

                # Draw and display the corners
                cv2.drawChessboardCorners(img, (corner_rows,corner_cols), corners, ret)
                write_name = chessboard_out_paths+str(idx)+'.jpg'
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
        # cv2.calibrateCamera(objectPoints, imagePoints, imageSize → retval, cameraMatrix, distCoeffs, rvecs, tvecs
        # distCoeffs: distortion coefficients  (k1, k2, p1, p2, [k3,[k4,k5,k6]])
        # rvecs: rotation vector
        # tvecs: translation vector
        # each k-th rotation vector together with the corresponding k-th translation vector
        # brings the calibration pattern from the model coordinate space (in which object points are specified)
        # to the world coordinate space,
        # that is, a real position of the calibration pattern in the k-th pattern view (k=0.. M -1).
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.img_size, None, None)
        print("self.mtx", self.mtx)
        # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
        dist_pickle = {}
        dist_pickle["mtx"] = self.mtx
        dist_pickle["dist"] = self.dist
        pickle.dump(dist_pickle, open(save_params_path, "wb"))

    def undistort(self, input_test, output_test, input_dist_pkl):
        dist_pickle = pickle.load(open(input_dist_pkl, 'rb'))
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
        print("mtx", mtx)
        print("dist", dist)
        # Test undistortion on an image
        img = cv2.imread(input_test)
        # undistort: Transforms an image to compensate for lens distortion.
        # cv2.undistort(src, cameraMatrix, distCoeffs[, dst[, newCameraMatrix]]) → dst
        dst = cv2.undistort(img, mtx, dist, None, mtx)
        cv2.imwrite(output_test,dst)
        return dst
        #dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        # Visualize undistortion
        # f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        # ax1.imshow(img)
        # ax1.set_title('Original Image', fontsize=30)
        # ax2.imshow(dst)
        # ax2.set_title('Undistorted Image', fontsize=30)

def main():
    chessboard_in_paths = '../camera_cal/*.jpg'
    chessboard_out_paths = '../output_images/camera_cal_out/'
    save_params_path = '../camera_calib_param/dist_pickle.p'
    chessboard_input_test = '../camera_cal/calibration1.jpg'
    chessboard_output_test = '../output_images/camera_cal_out/calibration1_undist.jpg'
    car_input_test = '../test_images/test2.jpg'
    car_output_test = '../output_images/test_image_out/test2_undist.jpg'
    input_dist_pkl = '../camera_calib_param/dist_pickle.p'

    corner_rows = 6
    corner_cols = 9

    calibTool = CameraCalibration()
    #calibTool.calc_param(chessboard_in_paths, chessboard_out_paths, corner_rows=corner_rows, corner_cols=corner_cols, save_params_path=save_params_path)
    # calibTool.undistort(chessboard_input_test, chessboard_output_test,input_dist_pkl)
    calibTool.undistort(car_input_test, car_output_test, input_dist_pkl)

if __name__ == "__main__":
    main()