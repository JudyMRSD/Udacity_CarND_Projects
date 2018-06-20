import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
from feature_util import FeatureUtil
from sklearn.svm import LinearSVC
import time
import pickle


# Hyper parameters
HOG_Color_Space = 'YUV'  # Can be RGB or YUV
HOG_Orient = 15  # HOG orientations
HOG_Pixel_Per_Cell = 8  # HOG pixels per cell
HOG_Cells_Per_Block = 2  # HOG cells per block

class DetectionPipeline:
    def __init__(self):
        self.feature_util = FeatureUtil(hog_orient = HOG_Orient,
                                        hog_pixel_per_cell = HOG_Pixel_Per_Cell,
                                        hog_cell_per_block = HOG_Cells_Per_Block,
                                        hog_color_space = HOG_Color_Space)

    def train_svm(self, data_folder):
        X_train, X_test, y_train, y_test = self.feature_util.prep_feature_dataset(data_folder)

        # Use a linear SVC
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        self.svc.fit(X_train, y_train)
        pickle.dump(self.svc, open("../Data/svc_model.p", "wb" ))

        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))



    def detect_image(self, image_path):
        # load a pre-trained svc model from a serialized (pickle) file
        #dist_pickle = pickle.load(open("../Data/svc_pickle.p", "rb"))
        svc = LinearSVC()
        svc_model = pickle.dumps(svc)

        orient =15
        pix_per_cell = 8
        cell_per_block = 2
        spatial_size = (32,32)
        hist_bins = 32

        img = mpimg.imread(image_path)
        bbox_scale = []
        # hyper parameter from https://github.com/TusharChugh/Vehicle-Detection-HOG/blob/master/src/vehicle-detection.ipynb
        scales = [1, 1.5, 2, 2.5, 3]
        ystarts = [400, 400, 450, 450, 460]
        ystops = [528, 550, 620, 650, 700]

        for scale, ystart, ystop in zip(scales, ystarts, ystops):
            out_img, bbox_list = self.feature_util.find_cars(img, svc_model, ystart, ystop, scale, orient, pix_per_cell,
                                                cell_per_block, spatial_size, hist_bins)
            if (len(bbox_list))>0:
                bbox_scale.extend(bbox_list)

        self.util.heat_map(img, bbox_scale)



def main():
    data_folder = "../Data/"

    dp = DetectionPipeline()
    image = '../Data/test_images/test4.jpg'
    dp.train_svm(data_folder)
    dp.detect_image(image)

    # pl.detect_video(video_name)


if __name__ == "__main__":

    main()