# adapted from Udacity course sample code section for color and graident
# Use color transforms, gradients, etc., to create a thresholded binary image.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
class Image_Process:
    def __init__(self):
        self.a = 0
    def sobel_thresh(self):
        if (self.sobel_flag == True):
            # Grayscale image
            # NOTE: we already saw that standard grayscaling lost color information for the lane lines
            # Explore gradients in other colors spaces / color channels to see what might work better
            self.gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
            # Sobel x
            sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
            abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
            scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
            # Threshold x gradient
            thresh_min = 20
            thresh_max = 255
            # self.sxbinary = np.zeros_like(scaled_sobel)
            self.sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    def hls_thresh(self):
        # Convert to HLS color space and separate the V channel
        hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS).astype(np.float)

        # Convert to HLS color space and separate the S channel
        # Note: img is the undistorted image
        hls = cv2.cvtColor(self.img, cv2.COLOR_RGB2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Threshold saturation channel
        s_thresh_min = 170
        s_thresh_max = 255
        if self.hls_saturation_flag == True:
            # s_binary = np.zeros_like(s_channel)
            self.s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

        if self.hls_lightness_flag == True:

            # Threshold lightness channel
            l_thresh_min = 40
            l_thresh_max = 255
            # l_binary = np.zeros_like(l_channel)
            self.l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    def stack_channel(self, sobel_flag, hls_saturation_flag, hls_lightness_flag):
        self.sobel_flag = sobel_flag
        self.hls_saturation_flag = hls_saturation_flag
        self.hls_lightness_flag = hls_lightness_flag
        self.img_h, self.img_w, _ = self.img.shape
        self.sxbinary = np.zeros((self.img_h, self.img_w))
        self.s_binary = np.zeros((self.img_h, self.img_w))
        self.l_binary = np.zeros((self.img_h, self.img_w))

        self.sobel_thresh()
        self.hls_thresh()
        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        self.color_binary = np.dstack((self.l_binary, self.sxbinary, self.s_binary)) * 255
        self.color_binary= self.color_binary.astype(np.uint8)



    def combine_thresh(self):
        # Combine the all the binary thresholds
        self.combined_binary = np.zeros_like(self.sxbinary)
        self.combined_binary[(self.s_binary == 1) & (self.l_binary == 1) | (self.sxbinary == 1)] = 1

        kernel = np.ones((5, 5), np.uint8)
        self.closing = cv2.morphologyEx(self.combined_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)


    def visualize(self, img, outdir=None, base=None):
        self.img = img
        print("visualize")

        # Plotting thresholded images
        f, ax = plt.subplots(2, 3, figsize=(20, 10))
        plt.tight_layout()

        self.stack_channel(sobel_flag=True, hls_saturation_flag=False, hls_lightness_flag=False)
        ax[0][0].set_title('sobel thresholds')
        ax[0][0].imshow(self.color_binary)
        self.stack_channel(sobel_flag=False, hls_saturation_flag=True, hls_lightness_flag=False)
        ax[0][1].set_title('hls saturation channel thresholds')
        ax[0][1].imshow(self.color_binary)
        self.stack_channel(sobel_flag=False, hls_saturation_flag=False, hls_lightness_flag=True)
        ax[0][2].set_title('hls lightness channel thresholds')
        ax[0][2].imshow(self.color_binary)
        self.stack_channel(sobel_flag=True, hls_saturation_flag=True, hls_lightness_flag=True)
        ax[1][0].set_title('stacked channels')  # remove effects from shadow
        ax[1][0].imshow(self.color_binary, cmap='gray')

        self.combine_thresh()
        ax[1][1].set_title('combined binary')  # remove effects from shadow
        ax[1][1].imshow(self.combined_binary)
        ax[1][2].set_title('after closing')  # remove effects from shadow
        self.closing *= 255
        ax[1][2].imshow(self.closing, cmap='gray')

        # plt.show()
        # cv2.imshow("self.closing", self.closing*255)
        # cv2.waitKey(0)
        if (outdir is not None) and (base is not None):
            plt.savefig(outdir + base + "channels.jpg")

def main():
    print("main")
    outdir = '../output_images/thresh_out/'
    input_img_path='../test_images/test5.jpg'
    image = mpimg.imread(input_img_path)
    base = os.path.basename(input_img_path)
    base = os.path.splitext(base)[0]

    pipeline = Image_Process()
    pipeline.visualize(image, outdir, base)

if __name__ == "__main__":
    main()