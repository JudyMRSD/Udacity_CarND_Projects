import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
class Image_Process:
    def __init__(self, outdir, base, img):
        self.outdir = outdir
        self.base = base
        self.img = img

    def sobel_thresh(self):
        if (self.sobel_flag == True):
            print("sobel")
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
        color_binary = np.dstack((self.l_binary, self.sxbinary, self.s_binary)) * 255
        color_binary= color_binary.astype(np.uint8)

        # Combine the all the binary thresholds
        combined_binary = np.zeros_like(self.sxbinary)
        combined_binary[(self.s_binary == 1) | (self.sxbinary == 1)] = 1

        combined_binary_l = np.zeros_like(self.sxbinary)
        combined_binary_l[(self.s_binary == 1) & (self.l_binary == 1) | (self.sxbinary == 1)] = 1

        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(combined_binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

        return color_binary, closing

    def visualize(self):

        # Plotting thresholded images
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
        plt.tight_layout()

        color_binary, closing = self.stack_channel(sobel_flag=True, hls_saturation_flag=False, hls_lightness_flag=False)
        ax1.set_title('sobel thresholds')

        ax1.imshow(color_binary)
        color_binary, closing = self.stack_channel(sobel_flag=False, hls_saturation_flag=True, hls_lightness_flag=False)
        ax2.set_title('hls saturation channel thresholds')
        ax2.imshow(color_binary)
        color_binary, closing = self.stack_channel(sobel_flag=False, hls_saturation_flag=False, hls_lightness_flag=True)
        ax3.set_title('hls lightness channel thresholds')
        ax3.imshow(color_binary)
        color_binary, closing = self.stack_channel(sobel_flag=True, hls_saturation_flag=True, hls_lightness_flag=True)
        ax4.set_title('l binary thresholds')  # remove effects from shadow
        ax4.imshow(color_binary)

        plt.savefig(self.outdir + self.base + "channels.jpg")
def main():
    outdir = '../output_images/thresh_out/'
    input_img_path='../test_images/test5.jpg'
    image = mpimg.imread(input_img_path)
    base = os.path.basename(input_img_path)
    base = os.path.splitext(base)[0]

    pipeline = Image_Process(outdir, base, image)
    pipeline.visualize()

if __name__ == "__main__":
    main()