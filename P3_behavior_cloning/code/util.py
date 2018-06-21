#
# visualize data: distribution of steering angles
import csv
import pandas as pd
import cv2

class DataUtil():
    def __init__(self):
        pass

    def parse_img_path(self, img_data_dir, paths):
        rgb_images = []
        # convert Panda to numpy array for speed
        for p in paths:
            file_name = p.split('/')[-1]
            full_path = img_data_dir + file_name
            bgr_img = cv2.imread(full_path)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb_img)
        return rgb_img

    def create_dataset(self, img_data_dir, ground_truth_path):
            # default header=0 and column names are inferred from the first line of the file
            ground_truth = pd.read_csv(ground_truth_path)

            center_paths = ground_truth['center'].values


            self.X_train = np.array(images)
            self.y_train = np.array(measurements)