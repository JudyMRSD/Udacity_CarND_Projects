import cv2
import numpy as np
import matplotlib.pyplot as plt


out_dir = "../output_images/thresh_out/"
class ImagePreprocess:
    def __init__(self):
        self.a = 0

    def plot_channels(self, color_space_imgs, color_space_list):

        plt.close('all')
        fig, ax = plt.subplots(len(color_space_list),3)
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
        plt.savefig(out_dir +"channels.jpg")

    # overlay binary mask on original image
    def plot_channel_thresh(self, img, img_channel_names, img_channel_list, thresh_list):
        fig, ax = plt.subplots(2, len(img_channel_list))
        for i, channel in enumerate(img_channel_list):
            ret, channel_binary = cv2.threshold(channel, thresh_list[i][1], thresh_list[i][1], cv2.THRESH_BINARY)
            ax[0][i].imshow(channel_binary, cmap='gray')
            ax[0][i].set_title(img_channel_names[i])
            channel_binary_mask = cv2.cvtColor(channel_binary, cv2.COLOR_GRAY2BGR)
            overlay = cv2.bitwise_and(img, channel_binary_mask)
            ax[1][i].imshow(overlay)
            ax[1][i].set_title(img_channel_names[i])

        plt.savefig(out_dir + "thresh.jpg")


    def all_color_spaces(self, input_img_path):
        color_space_list = ['HLS', 'HVS', 'RGB', 'gray']

        img = cv2.imread(input_img_path)
        hsl_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        rgb_img = img
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        color_space_imgs = [hsl_img, hsv_img, rgb_img, gray_img]
        self.plot_channels(color_space_imgs, color_space_list)

        rgb_r = rgb_img[:,:,0]
        hvs_s = hsv_img[:,:,2]
        hsl_s = hsl_img[:,:,1]

        img_channel_names = ["gray", "R in RGB", "S in HSV","S in HSL"]
        img_channel_list = [gray_img, rgb_r, hvs_s, hsl_s]
        thresh_list = [(180, 255), (200, 255), (90, 255), (70, 255)]
        self.plot_channel_thresh(img, img_channel_names, img_channel_list, thresh_list)



def main():
    imgProcessor = ImagePreprocess()
    input_img = '../output_images/test_image_out/test2_undist.jpg'

    print("binary")
    #imgProcessor.binary_HSV(input_img)
    #imgProcessor.binary_HLS(input_img)
    imgProcessor.all_color_spaces(input_img)

if __name__ == "__main__":
    main()