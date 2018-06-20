import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
from feature_util import FeatureUtil
from img_util import ImgUtil

from sklearn.svm import LinearSVC
import time
import pickle


# Hyper parameters
HOG_Color_Space = 'YUV'  # Can be RGB or YUV
HOG_Orient = 15  # HOG orientations
HOG_Pixel_Per_Cell = 8  # HOG pixels per cell
HOG_Cells_Per_Block = 2  # HOG cells per block
Thresh_Heatmap = 1

Svc_Pickle =  "../Data/model/svc_model.p"
Writeup_Imgs_Dir = "../Data/experiment_outputs/"
class DetectionPipeline:
    def __init__(self):
        self.feature_util = FeatureUtil(hog_orient = HOG_Orient,
                                        hog_pixel_per_cell = HOG_Pixel_Per_Cell,
                                        hog_cell_per_block = HOG_Cells_Per_Block,
                                        hog_color_space = HOG_Color_Space)
        self.imgUtil = ImgUtil()
    def train_svm(self, data_folder):
        X_train, X_test, y_train, y_test = self.feature_util.prep_feature_dataset(data_folder)

        # Use a linear SVC
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        self.svc.fit(X_train, y_train)
        # TODO: dump X_scalar to pickle
        pickle.dump(self.svc, open(Svc_Pickle, "wb" ))

        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))



    def detect_image(self, image_path):
        # load a pre-trained svc model from a serialized (pickle) file
        #dist_pickle = pickle.load(open("../Data/svc_pickle.p", "rb"))

        svc_model = pickle.load(open(Svc_Pickle, "rb" ))

        img = mpimg.imread(image_path)
        bbox_scale = []
        # hyper parameter from https://github.com/TusharChugh/Vehicle-Detection-HOG/blob/master/src/vehicle-detection.ipynb
        scales = [1, 1.5, 2, 2.5, 3]
        ystarts = [400, 400, 450, 450, 460]
        ystops = [528, 550, 620, 650, 700]

        for scale, ystart, ystop in zip(scales, ystarts, ystops):
            out_img, bbox_list = self.feature_util.find_cars(img, svc_model, ystart, ystop, scale)
            if (len(bbox_list))>0:
                bbox_scale.extend(bbox_list)

        self.imgUtil.heat_map(img, bbox_scale, Writeup_Imgs_Dir, Thresh_Heatmap)

def main():
    data_folder = "../Data/"

    dp = DetectionPipeline()
    image = '../Data/test_images/test4.jpg'
    dp.train_svm(data_folder)
    dp.detect_image(image)
    # pl.detect_video(video_name)

if __name__ == "__main__":

    main()