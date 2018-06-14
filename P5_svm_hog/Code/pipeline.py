from utility import Utility
import glob
import numpy as np
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class Pipeline:
    def __init__(self):
        self.util = Utility()

    def train_svm(self, data_folder):
        # input: data_folder
        # output: SVM model
        # steps:
        # 1. prep_feature_dataset (train and test set (HOG features with label))
        # 2. train
        # 3. save model
        # Read in cars and notcars

        # todo: unify the parameters with the ones used for training svm
        cars = glob.glob(data_folder + "train_test_data/vehicles/*/*.png")
        notcars = glob.glob(data_folder + "train_test_data/non-vehicles/*/*.png")

        # Reduce the sample size because
        # The quiz evaluator times out after 13s of CPU time
        sample_size = 500
        cars = cars[0:sample_size]
        notcars = notcars[0:sample_size]
        color_space = 'RGB'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        orient = 15  # HOG orientations
        pix_per_cell = 8  # HOG pixels per cell
        cell_per_block = 2  # HOG cells per block
        hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        spatial_size = (32, 32)  # Spatial binning dimensions
        hist_bins = 32  # Number of histogram bins
        spatial_feat = True  # Spatial features on or off
        hist_feat = True  # Histogram features on or off
        hog_feat = True  # HOG features on or off


        car_features = self.util.extract_features(cars, color_space=color_space,
                                        spatial_size=spatial_size, hist_bins=hist_bins,
                                        orient=orient, pix_per_cell=pix_per_cell,
                                        cell_per_block=cell_per_block,
                                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                                        hist_feat=hist_feat, hog_feat=hog_feat)
        notcar_features = self.util.extract_features(notcars, color_space=color_space,
                                           spatial_size=spatial_size, hist_bins=hist_bins,
                                           orient=orient, pix_per_cell=pix_per_cell,
                                           cell_per_block=cell_per_block,
                                           hog_channel=hog_channel, spatial_feat=spatial_feat,
                                           hist_feat=hist_feat, hog_feat=hog_feat)

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=rand_state)

        # Fit a per-column scaler
        self.X_scaler = StandardScaler().fit(X_train)
        # Apply the scaler to X
        X_train = self.X_scaler.transform(X_train)
        X_test = self.X_scaler.transform(X_test)

        print('Using:', orient, 'orientations', pix_per_cell,
              'pixels per cell and', cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        self.svc = LinearSVC()
        # Check the training time for the SVC
        t = time.time()
        self.svc.fit(X_train, y_train)
        # pickle.dump(svc, open( "../Data/svc_model.p", "wb" ) )

        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')
        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))

    # Define a single function that can extract features using hog sub-sampling and make predictions
    def find_cars(self, img, ystart, ystop, scale, orient, pix_per_cell, cell_per_block, spatial_size,
                  hist_bins):
        # input: saved SVM model
        # output: bbox detections
        # steps:
        # 1. load SVM model
        # 2. sliding_window
        # 3. refine_bbox
        draw_img = np.copy(img)
        # img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        # ctrans_tosearch = self.util.convert_color(img_tosearch, conv='RGB2YCrCb')
        #if scale != 1:
        #    imshape = ctrans_tosearch.shape
        #    ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = img_tosearch[:, :, 0]
        ch2 = img_tosearch[:, :, 1]
        ch3 = img_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
        nfeat_per_block = orient * cell_per_block ** 2

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

        # Compute individual channel HOG features for the entire image
        hog1 = self.util.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = self.util.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = self.util.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(img_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = self.util.bin_spatial(subimg, size=spatial_size)
                hist_features = self.util.color_hist(subimg, nbins=hist_bins)

                # Scale features and make a prediction
                test_stack = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
                test_features = self.X_scaler.transform(test_stack)

                # test_features = self.X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = self.svc.predict(test_features)
                print("test_prediction", test_prediction)
                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)

                    win_draw = np.int(window * scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
        cv2.imshow('detection', draw_img)
        cv2.waitKey(0)

        return draw_img

    def detect_image(self, image_path):
        # pl.train_svm(data_folder)
        ystart = 400
        ystop = 500
        scale = 1.5
        # load a pre-trained svc model from a serialized (pickle) file
        #dist_pickle = pickle.load(open("../Data/svc_pickle.p", "rb"))
        #svc = LinearSVC()
        #svc_model = pickle.dumps(svc)
        # get attribu../Datates of our svc object
        # svc = dist_pickle["svc"]
        #X_scaler = dist_pickle["scaler"]
        orient =15
        pix_per_cell = 8
        cell_per_block = 2
        spatial_size = (32,32)
        hist_bins = 32

        img = mpimg.imread(image_path)
        out_img = self.find_cars(img, ystart, ystop, scale, orient, pix_per_cell, cell_per_block,
                               spatial_size,
                               hist_bins)
        plt.imshow(out_img)
        plt.show()
        # plt.savefig("result.jpg")

    def detect_video(self, video_name):
        # run detect_img on a video
        pass

def main():
    data_folder = "../Data/"

    pl = Pipeline()
    image = '../Data/test_images/test4.jpg'
    pl.train_svm(data_folder)
    pl.detect_image(image)

    # pl.detect_video(video_name)


if __name__ == "__main__":

    main()



