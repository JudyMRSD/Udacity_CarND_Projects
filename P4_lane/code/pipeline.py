from utility.cameraCalib import CameraCalibration
from utility.imgProcess import Image_Process
from utility.perspectiveTransform import Perspective
from utility.laneBoundary import Boundary
import os
import numpy as np
import cv2
import tqdm
import pickle


class Pipeline:
    def __init__(self):
        self.calibTool = CameraCalibration()
        self.threshTool = Image_Process()
        self.perspectiveTool = Perspective()
        self.boundaryTool = Boundary()

    def check_paths(self, data_dir):
        self.chessboard_in_dir = data_dir + 'camera_cal/'
        self.chessboard_out_dir = data_dir + 'camera_cal_out/'
        # directories for chessboard images used in calibration and place to save parameters after calibration
        self.save_params_path = data_dir + 'camera_calib_param/dist_pickle.p'
        # check correctness of folders paths
        assert (os.path.exists(self.chessboard_in_dir)), "ERROR: Chessboard_In_Dir does not exist"
        assert (os.path.exists(self.chessboard_out_dir)), "ERROR: Chessboard_Out_Dir does not exist"

    def process_single_img(self, idx, input_img, out_dir = None):
        '''
        process single image, save intermediate results if output_img_basename is given
        :param idx: frame index
        :param input_img: input frame read from video
        :param out_dir: directory to save intermediate visualizations
        '''
        # following Steps are in the order that's listed in README.me
        # Step 2: Apply a distortion correction to raw images.
        undistort_front = cv2.undistort(input_img, self.camera_matrix, self.distorsion_coefficient, None, self.camera_matrix)
        if (out_dir):
            cv2.imwrite(out_dir + "undistort_front.jpg", undistort_front)
        # Step 3: Use color transforms, gradients, etc., to create a thresholded binary image
        binary_front_img = self.threshTool.visualize(undistort_front, out_dir)
        if (idx==0):
            # Step 4: Apply a perspective transform to rectify binary image ("birds-eye view").
            # to_bird_matrix, to_front_matrix only need to be calculated on the first frame
            birdeye_binary, self.to_bird_matrix, self.to_front_matrix = self.perspectiveTool.warp_front_to_birdeye(
                                                                    binary_front_img, out_dir)

            self.boundaryTool.histogram_peaks(out_dir, birdeye_binary)

        else:
            # Step 4: use previously calculated matrix to warp image
            h, w = binary_front_img.shape[0:2]
            birdeye_binary = cv2.warpPerspective(binary_front_img, self.to_bird_matrix, (w, h))

        # Step 5: Detect lane pixels and fit to find the lane boundary as f(y)
        # Determine curvature of the lane and vehicle position with respect to center
        self.boundaryTool.fit_lane(birdeye_binary, idx)
        # Step 6-7: Determine the curvature and vehicle position, and warp back to original image
        result_imgself = self.boundaryTool.visualize_lane(out_dir, input_img, self.to_front_matrix, blend_alpha=0.5)
        return result_imgself

    def process_video(self, in_video, out_video, calibrate):
        '''
        Process video by detecting lanes frame by frame
        :param in_video: input video path
        :param out_video: output video path
        :param calibrate: flag to indicate whether calibration is needed
        '''
        # Step 1 : Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
        if not os.path.exists(self.chessboard_out_dir) or calibrate:
            self.calibTool.calc_param(self.chessboard_in_dir, self.chessboard_out_dir, 6, 9, self.save_params_path)

        cap = cv2.VideoCapture(in_video)
        # get properties of input video, so the output video can match these properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter_fourcc(*'mp4v')
        # process videos frame by frame, from the first to the last frame
        frame_indices = np.arange(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)

        out_video = cv2.VideoWriter(out_video, writer, fps, (frame_width, frame_height))  # run Mac

        dist_pickle = pickle.load(open(self.save_params_path, 'rb'))
        self.camera_matrix = dist_pickle["camera_matrix"]
        self.distorsion_coefficient = dist_pickle["distorsion_coefficient"]

        # use tqdm progress bar
        for i in tqdm.tqdm(frame_indices):
            success, in_frame = cap.read()
            if success:
                # stop the program after finding the frame where ball was released, and record the frame index
                out_frame = self.process_single_img(i, in_frame)
                out_video.write(out_frame)
        out_video.release()


def main():
    # Following are directory paths and parameter values that could be changed if running on a different dataset
    data_dir = '../data/'
    in_video = data_dir+"video/input_video/project_video.mp4"
    out_video = data_dir+"video/output_video/project_video_out.mp4"

    pl = Pipeline()
    pl.check_paths(data_dir)

    #input and output video directory
    # Step 2-7 : process each frame in the video
    pl.process_video(in_video, out_video, calibrate=False)


if __name__ == "__main__":
    main()