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


class ParamUtil:
    def __init__(self):
        self.imgUtil = ImgUtil()
    def hog_param_vis(self, train_data_folder, out_dir):
        vehi_image_name = glob.glob(train_data_folder + "vehicles/*/*.png")[0]
        no_vehi_img_name = glob.glob(train_data_folder + "non-vehicles/*/*.png")[0]
        vehicle_image = cv2.imread(vehi_image_name)
        non_vehi_image = cv2.imread(no_vehi_img_name)
        # visualize hog for different images
        hog_orient_list = [9, 15]
        hog_pixel_per_cell_list = [8, 16]
        hog_cell_per_block_list = [1, 2]
        hog_color_space_list = ['RGB', 'HSV', 'YUV']

        # num_parameter_comb = len(hog_orient_list) * len(hog_pixel_per_cell_list)\
        #                      * len(hog_cell_per_block_list) * len(hog_color_space_list)
        # 4: 1 original image + 3 channels hog for image
        # 2: car, no car

        for ori in hog_orient_list:
            for pixPcell in hog_pixel_per_cell_list:
                for cellPblock in hog_cell_per_block_list:
                    for colorS in hog_color_space_list:
                        plt.close('all')
                        fig, ax = plt.subplots(2, 4, figsize=(24, 9))
                        self.hog_single_img_vis(vehicle_image,"car", ax, 0,
                                                ori, pixPcell, cellPblock, colorS)
                        self.hog_single_img_vis(non_vehi_image,"no car", ax, 1,
                                                ori, pixPcell, cellPblock, colorS)

                        plt.tight_layout()
                        param_file_name = colorS +"_ori_"+str(ori) + "_pixPcell_"+str(pixPcell)+ "_cellPblock_"+ str(cellPblock)
                        plt.savefig(out_dir + "hog_params/"+ param_file_name+".jpg")


    def hog_single_img_vis(self, image, image_class, ax, ax_row, hog_orient, hog_pixel_per_cell,
                           hog_cell_per_block, hog_color_space):
        image = self.imgUtil.convert_color(image, hog_color_space)
        channels = image.shape[-1]

        ax[ax_row][0].imshow(image)
        ax[ax_row][0].set_title(image_class)

        for i in range (channels):
            # feature_vector=False means feature is not changed to 1d vector using .ravel()
            _, hog_img = hog(image[:,:,i], orientations= hog_orient,
                                     pixels_per_cell=(hog_pixel_per_cell, hog_pixel_per_cell),
                                     cells_per_block=(hog_cell_per_block, hog_cell_per_block),
                                     block_norm='L2-Hys',
                                     transform_sqrt=True,
                                     visualise=True, feature_vector=False)
            ax[ax_row][i+1].imshow(hog_img, cmap='gray')
            ax[ax_row][i+1].set_title(hog_color_space[i])
