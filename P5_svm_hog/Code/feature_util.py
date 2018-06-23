import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.preprocessing import StandardScaler
from img_util import ImgUtil
import glob
from sklearn.model_selection import train_test_split
import tqdm


numExample = -1 # -1 use all images


class FeatureUtil:
    def __init__(self, hog_orient, hog_pixel_per_cell,
                 hog_cell_per_block, hog_color_space='RGB'):

        self.a = 0
        self.imgUtil = ImgUtil()
        self.hog_orient = hog_orient
        self.hog_pixel_per_cell = hog_pixel_per_cell
        self.hog_cell_per_block = hog_cell_per_block
        self.hog_color_space = hog_color_space

    def hog_single_img(self, image):
        image = self.imgUtil.convert_color(image, self.hog_color_space)
        hog_features = []
        if (len(image.shape) < 3):
            print("error: input must have 3 channels")
        w, h, channels = image.shape
        num_dim_hog_single_channel = 0
        for c in range(channels):
            # feature_vector=False means feature is not changed to 1d vector using .ravel()
            hog_channel = hog(image[:,:,c], orientations=self.hog_orient,
                                     pixels_per_cell=(self.hog_pixel_per_cell, self.hog_pixel_per_cell),
                                     cells_per_block=(self.hog_cell_per_block, self.hog_cell_per_block),
                                     block_norm='L2-Hys',
                                     transform_sqrt=True,
                                     visualise=False, feature_vector=False)
            num_dim_hog_single_channel = hog_channel.ndim
            hog_channel = np.expand_dims(hog_channel, axis = num_dim_hog_single_channel)
            hog_features.append(hog_channel)
        hog_features_all = np.concatenate([hog_features[i] for i in range(channels)], axis=num_dim_hog_single_channel) # (7, 7, 2, 2, 15)
        # print("hog_features_all.shape", hog_features_all.shape) # (7, 7, 6, 2, 15)
        return hog_features_all

    # use all channels to extract HOG features
    def hog_multiple_imgs(self, imgs):
        features = []
        # Iterate through list of images
        img_idx = np.arange(0, len(imgs))
        for i in tqdm.tqdm(img_idx):
            file_features = []
            # Read in each one by one
            image = cv2.imread(imgs[i])
            hog_features = self.hog_single_img(image)
            hog_features = np.ravel(hog_features)
            features.append(hog_features)
        return features

    # create test and train data for svm
    def prep_feature_dataset(self, data_folder):
        cars = glob.glob(data_folder + "vehicles/*/*.png")
        notcars = glob.glob(data_folder + "non-vehicles/*/*.png")
        cars = cars[:numExample]
        notcars = notcars[:numExample]
        print("extract car features")
        car_features = self.hog_multiple_imgs(cars)
        print("extract notcars features")
        notcar_features = self.hog_multiple_imgs(notcars)

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        # train_test_split,   shuffle :(default=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

        # Fit a per-column scaler
        X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = X_scaler.transform(X_train)
        X_test = X_scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, X_scaler

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, svc_model, ystart, ystop, X_scaler, scale):
        # input: saved SVM model
        # output: bbox detections
        # steps:
        # 1. load SVM model
        # 2. sliding_window
        # 3. refine_bbox
        draw_img = np.copy(img)
        # img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        # downscale image is the same as using larger window
        if scale != 1:
            imshape = img_tosearch.shape
            img_tosearch = cv2.resize(img_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        height, width, channels = img_tosearch.shape
        img_tosearch = cv2.resize(img_tosearch, (int(width/ scale), int(height / scale)))

        # Define blocks and steps as above
        nxblocks = (width // self.hog_pixel_per_cell) - self.hog_pixel_per_cell + 1
        nyblocks = (height // self.hog_pixel_per_cell) - self.hog_pixel_per_cell + 1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        win_draw = window * scale

        nblocks_per_window = (window // self.hog_pixel_per_cell) - self.hog_cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
        # nyblocks   x   nxblocks   x    numcell per block  x numcell per block   x  orientations
        hog_features = self.hog_single_img(img_tosearch)

        bbox_list = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_window = hog_features[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window, :]

                xleft = xpos * self.hog_pixel_per_cell
                ytop = ypos * self.hog_pixel_per_cell

                test_features = X_scaler.transform(hog_window.reshape(1,-1))
                test_prediction = svc_model.predict(test_features)

                if test_prediction == 1:
                    xmin = int(xleft * scale)
                    ymin = int(ytop * scale)+ ystart
                    xmax = xmin + win_draw
                    ymax = ymin + win_draw

                    cv2.rectangle(draw_img, (xmin, ymin),(xmax, ymax), (0, 0, 255), 6)
                    bbox_list.append(((xmin, ymin), (xmax, ymax)))
        return draw_img, bbox_list













