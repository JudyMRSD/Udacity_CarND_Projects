import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from feature_util import FeatureUtil
from img_util import ImgUtil
from experiment_param import ParamUtil
import tqdm
from sklearn.svm import LinearSVC
import time
import pickle
import os

# Hyper parameters
HOG_Color_Space = 'YUV'  # Can be RGB or YUV
HOG_Orient = 15  # HOG orientations
HOG_Pixel_Per_Cell = 8  # HOG pixels per cell
HOG_Cells_Per_Block = 2  # HOG cells per block
Thresh_Heatmap = 1

# number of history frames to keep for more rubust heatmap
Hist_Len = 15

Svc_Pickle =  "../Data/model/svc_model.p"
Writeup_Imgs_Dir = "../Data/output_images/"
Video_Folder = "../Data/input_videos/"

class DetectionPipeline:
    def __init__(self):
        self.feature_util = FeatureUtil(hog_orient = HOG_Orient,
                                        hog_pixel_per_cell = HOG_Pixel_Per_Cell,
                                        hog_cell_per_block = HOG_Cells_Per_Block,
                                        hog_color_space = HOG_Color_Space)
        self.imgUtil = ImgUtil()
        self.paramUtil = ParamUtil()
        self.model_dict = {}
        self.bbox_hist = []

    def train_svm(self, data_folder):
        X_train, X_test, y_train, y_test, X_scaler  = self.feature_util.prep_feature_dataset(data_folder)

        # Use a linear SVC
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        self.svc.fit(X_train, y_train)
        self.model_dict['svc_model'] =  self.svc
        self.model_dict['X_scaler'] = X_scaler
        # include  time as part of model name to avoid overwrite good model
        pickle.dump(self.model_dict, open(Svc_Pickle, "wb" ))

        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))


    def detect_image(self, img, img_idx, verbose = False):
        # load a pre-trained svc model from a serialized (pickle) file
        #dist_pickle = pickle.load(open("../Data/svc_pickle.p", "rb"))
        svc_model_dict = pickle.load(open(Svc_Pickle, "rb" ))
        svc_model = svc_model_dict['svc_model']
        X_scaler = svc_model_dict['X_scaler']

        bbox_scale = []
        # hyper parameter from https://github.com/TusharChugh/Vehicle-Detection-HOG/blob/master/src/vehicle-detection.ipynb
        scales = [1, 1.5, 2, 2.5, 3]
        ystarts = [400, 400, 450, 450, 460]
        ystops = [528, 550, 620, 650, 700]

        for scale, ystart, ystop in zip(scales, ystarts, ystops):
            out_img, bbox_list = self.feature_util.find_cars(img, svc_model, ystart, ystop, X_scaler, scale)
            if (len(bbox_list))>0:
                bbox_scale.extend(bbox_list)

        self.bbox_hist.append(bbox_scale)
        draw_img = self.imgUtil.heat_map(img, self.bbox_hist, Writeup_Imgs_Dir, Thresh_Heatmap, verbose)
        if (verbose):
            cv2.imwrite(Writeup_Imgs_Dir + str(img_idx)+"_bbox_heatmap.jpg", draw_img)
        # only keep history bbox in recent frames
        if (len(self.bbox_hist) > Hist_Len):
            self.bbox_hist = self.bbox_hist[1:]
        return draw_img

    def detect_video(self, video_path):

        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_ids = np.arange(1, video_length)

        video_out_name = Writeup_Imgs_Dir + os.path.splitext(os.path.basename(video_path))[0]
        video_writer = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_wid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_video = cv2.VideoWriter(video_out_name, video_writer, fps, (frame_wid, frame_height))

        for i in tqdm.tqdm(frame_ids):
            success, frame = cap.read()
            if success:
                draw_img = self.detect_image(frame, img_idx=i, verbose = False)
                out_video.write(draw_img)
            else:
                print("ERROR: failed to read frame "+str(i))
        cap.release()
        out_video.release()


def main():
    data_folder = "../Data/"
    # video_name = Video_Folder + "test_video.mp4"
    video_name = Video_Folder + "project_video.mp4"
    train_data_folder = "../Data/train_test_data/"

    dp = DetectionPipeline()

    #dp.paramUtil.hog_param_vis(train_data_folder, Writeup_Imgs_Dir)
    # image_path = '../Data/test_images/test4.jpg'
    # TODO: save scalar too
    # dp.train_svm(train_data_folder)
    # img = mpimg.imread(image_path)
    # dp.detect_image(img, 0,verbose=True)
    dp.detect_video(video_name)

if __name__ == "__main__":

    main()