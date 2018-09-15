import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


out_dir = "../output_images/thresh_out/"
class ColorSpace:

    def plot_channels(self, base, color_space_imgs, color_space_list):

        plt.close('all')
        fig, ax = plt.subplots(len(color_space_list),3, figsize=(16, 9))
        plt.tight_layout()

        for i, color_space in enumerate(color_space_list):
            img = color_space_imgs[i]
            print("img.shape", img.shape)

            if (color_space=='gray'):
                channel_names = 'gray'
                ax[i][0].imshow(img, cmap='gray')
                ax[i][0].set_title("gray ")
            else:
                num_channels = img.shape[2]
                channel_names = list(color_space)
                for j in range(0, num_channels):
                    channel = img[:,:,j]
                    ax[i][j].imshow(channel, cmap = 'gray')
                    ax[i][j].set_title(color_space+", channel = " + channel_names[j])
        plt.savefig(out_dir + base+"channels.jpg")

    # overlay binary mask on original image
    def plot_channel_thresh(self, base, img_channel_names, img_channel_list, thresh_list):
        fig, ax = plt.subplots(2, len(img_channel_list), figsize=(16, 9))

        for i, channel in enumerate(img_channel_list):
            print("thresh_list[i][1], thresh_list[i][1]", thresh_list[i][0], thresh_list[i][1])
            ret, channel_binary = cv2.threshold(channel, thresh_list[i][0], thresh_list[i][1], cv2.THRESH_BINARY)

            ax[0][i].imshow(channel, cmap='gray')
            ax[0][i].set_title(img_channel_names[i])

            ax[1][i].imshow(channel_binary, cmap='gray')
            ax[1][i].set_title("thresh " +img_channel_names[i])


        plt.savefig(out_dir + base+"thresh.jpg")


    def all_color_spaces(self, input_img_path):
        color_space_list = ['HLS', 'HVS', 'BGR', 'gray']

        base = os.path.basename(input_img_path)
        base = os.path.splitext(base)[0]

        img = cv2.imread(input_img_path)
        if (img is None):
            print("bug: invalid image path")
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        brg_img = img
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        color_space_imgs = [hls_img, hsv_img, brg_img, gray_img]
        self.plot_channels(base, color_space_imgs, color_space_list)

        bgr_r = brg_img[:,:,2] # opencv is in bgr order
        hsv_s = hsv_img[:,:,1]
        hls_s = hls_img[:,:,2]

        img_channel_names = ["gray", "R in RGB", "S in HSV","S in HLS"]
        img_channel_list = [gray_img, bgr_r, hsv_s, hls_s]
        thresh_list = [(180, 255), (200, 255), (90, 255), (70, 255)]
        self.plot_channel_thresh(base, img_channel_names, img_channel_list, thresh_list)

def main():
    colorSpace_test = ColorSpace()
    # input_img = '../test_images/test6.jpg'
    # colorSpace_test.all_color_spaces(input_img)

    # input_img = '../test_images/test4.jpg'
    # colorSpace_test.all_color_spaces(input_img)

    input_img = '../test_images/test5.jpg'
    colorSpace_test.all_color_spaces(input_img)


if __name__ == "__main__":
    main()