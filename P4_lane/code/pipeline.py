from cameraCalib import CameraCalibration
from img_process import Image_Process
from perspective_transform import Perspective
from lane_boundary import Boundary
import os
import numpy as np
import cv2
import matplotlib.image as mpimg
import tqdm
import pickle



# Flag to run on MacOS or Ubuntu for video processing
Mac = True # False for Ubuntu system, different video format

# Following are directories global to pipeline.py
Data_Dir = '../data/'
# directories for chessboard images used in calibration and place to save parameters after calibration
Chessboard_In_Dir = Data_Dir+'camera_cal/'
Chessboard_Out_Dir = Data_Dir+'output_images/camera_cal_out/'
Save_Params_Path = Data_Dir+'camera_calib_param/dist_pickle.p'
# images used for testing the undistort performance on chessboard and car
Chessboard_Input_Test = Data_Dir+'camera_cal/2.jpg'
Chessboard_Output_Test = Data_Dir+'output_images/undistort/calibration2_undist.jpg'
Car_Input_Test = Data_Dir+'test_images/test2.jpg'
Car_Output_Test = Data_Dir+'output_images/undistort/calibration2_undistort.jpg'
# Output directory to save thresholded images
outdir = Data_Dir+'pipeline_out/'
input_img_path = Data_Dir+'test_images/test5.jpg'
# Points needed to warp images to bird eye view, these are hand selected points
# that can calculate a mapping that doesn't change as long as the intrinsics and
# extrinsics doesn't change
# (a1, b1), (a2,b2) maps to (0,0) and (w,0) in bird eye view
a1, b1, a2, b2 =  550, 460, 730, 460

# Camera calibration params for checker board
Corner_Rows = 6
Corner_Cols = 9


class Pipeline:
    def __init__(self):
        self.calibTool = CameraCalibration()
        self.threshTool = Image_Process()
        self.perspectiveTool = Perspective()
        self.boundaryTool = Boundary()

    def check_paths(self):
        # check correctness of file paths
        assert (os.path.exists(Chessboard_In_Dir)), "ERROR: Chessboard_In_Dir does not exist"
        assert (os.path.exists(Chessboard_Out_Dir)), "ERROR: Chessboard_Out_Dir does not exist"
        assert (os.path.exists(Save_Params_Path)), "ERROR: Save_Params_Path does not exist"
        assert (os.path.exists(Chessboard_Output_Test)), "ERROR: Chessboard_Output_Test does not exist"
        assert (os.path.exists(Car_Input_Test)), "ERROR: Car_Input_Test does not exist"


    # process single image, save intermediate results if output_img_basename is given
    def process_single_img(self, idx, input_img, out_dir = None):
        # Step 2: Apply a distortion correction to raw images.
        undistort_front = cv2.undistort(input_img, self.camera_matrix, self.distorsion_coefficient, None, self.camera_matrix)
        if (out_dir):
            print("write", outdir+"undistort_front.jpg")

            cv2.imwrite(outdir+"undistort_front.jpg",undistort_front)
        # Step 3: Use color transforms, gradients, etc., to create a thresholded binary image
        binary_front_img = self.threshTool.visualize(undistort_front, outdir)
        if (idx==1):
            # Step 4: Apply a perspective transform to rectify binary image ("birds-eye view").
            # to_bird_matrix, to_front_matrix only need to be calculated on the first frame
            birdeye_binary, self.to_bird_matrix, self.to_front_matrix = self.perspectiveTool.warp_front_to_birdeye(
                                                                    a1, b1, a2, b2,
                                                                    binary_front_img, out_dir)
            # Step 5: Detect lane pixels and fit to find the lane boundary as f(y)
            # Determine curvature of the lane and vehicle position with respect to center
            self.boundaryTool.histogram_peaks(outdir, birdeye_binary)
            self.boundaryTool.slide_window()
        else:
            # Step 4: use previously calculated matrix to warp image
            h, w = binary_front_img.shape[0:2]
            birdeye_binary = cv2.warpPerspective(binary_front_img, self.to_bird_matrix, (w, h))
            # Step 5: only search in a margin around the previous line position l
            # Detect lane and warp the detected lane boundaries back onto the original image
            self.boundaryTool.fit_use_prev(birdeye_binary)
        # Step 6-8: Determine the curvature and vehicle position, and warp back to original image
        result_imgself = self.boundaryTool.visualize_lane(outdir, input_img, self.to_front_matrix, blend_alpha=0.5)
        return result_imgself

    def process_video(self, in_video, out_video):


        '''
        Process one video, write the frame ID for when the ball was released to the log file
        :param video_path: path to the video of a baseball pitch
        '''
        cap = cv2.VideoCapture(in_video)
        # get properties of input video, so the output video can match these properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = cv2.VideoWriter_fourcc(*'mp4v')
        # process videos frame by frame, from the first to the last frame
        frame_indices = np.arange(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        if Mac:
            out_video = cv2.VideoWriter(out_video, writer, fps, (frame_width, frame_height))  # run Mac
        else:
            out_video = cv2.VideoWriter(out_video, 0x00000021, fps, (frame_width, frame_height))  # run on Ubuntu

        dist_pickle = pickle.load(open(Save_Params_Path, 'rb'))
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

    pl = Pipeline()
    pl.check_paths()
    # Step 1 : Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    # pl.calibTool.calc_param(Chessboard_In_Dir + "*.jpg", Chessboard_Out_Dir, corner_rows=Corner_Rows,
    #                           corner_cols=Corner_Cols, save_params_path=Save_Params_Path)

    dist_pickle = pickle.load(open(Save_Params_Path, 'rb'))
    pl.camera_matrix = dist_pickle["camera_matrix"]
    pl.distorsion_coefficient = dist_pickle["distorsion_coefficient"]
    input_img = cv2.imread(Car_Input_Test)
    pl.process_single_img(1, input_img, out_dir=Data_Dir+"pipeline_out/")
    # input and output video directory
    #in_video = Data_Dir+"video/input_video/project_video.mp4"
    #out_video = Data_Dir+"video/input_video/project_video_out.mp4"
    #pl.process_video(in_video, out_video)


if __name__ == "__main__":
    main()