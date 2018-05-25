import cv2
import matplotlib.pyplot as plt
import numpy as np


# Used code from Udacity online course 18 :  Detect lane pixels and fit to find the lane boundary
class Boundary():
    def __init__(self):
        self.a  = 0
    def histogram_peaks(self, outdir, img):
        self.img = img
        print("img shape", self.img.shape)
        #img = img/255
        # take a histogram along all the columns in the lower half of the image
        histogram = np.sum(self.img[self.img.shape[0] // 2:, :], axis=0)
        print("histogram shape", histogram.shape)
        plt.plot(histogram)
        plt.xlabel('Counts')
        plt.ylabel('Pixel Positions')
        plt.savefig(outdir + "channels.jpg")
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = int(histogram.shape[0] // 2)
        self.leftx_base = np.argmax(histogram[:midpoint])
        self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        print(histogram.shape[0],midpoint,self.leftx_base, self.rightx_base)
        plt.close()

    def slid_window(self):
        self.img_h, self.img_w = self.img.shape[0:2]
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows (image height divided by number of sliding windows)
        window_height = np.int(self.img_h // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.img.nonzero()
        self.nonzeroy = np.array(nonzero[0]) # y coordinate of non zero pixel
        self.nonzerox = np.array(nonzero[1]) # x coordinate of non zero pixel
        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []
        # Step through the windows one by one
        # Draw the windows on the visualization image
        self.out_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.img_h - (window + 1) * window_height
            win_y_high = self.img_h - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            cv2.rectangle(self.out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                          (0, 255, 0), 2)
            cv2.rectangle(self.out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                          (0, 255, 0), 2)

            # Identify x coordinate for nonzero pixels in the current window
            # good_right_inds[i] == 1 if ith point in nonzero points is inside the window
            good_left_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
                              (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((self.nonzeroy >= win_y_low) & (self.nonzeroy < win_y_high) &
                               (self.nonzerox >= win_xright_low) & (self.nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            self.left_lane_inds.append(good_left_inds)
            self.right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(self.nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(self.nonzerox[good_right_inds]))

        # Concatenate the arrays of indices:  left_lane_inds  num window rows, each row has num valid pixels in the window
        # turn this into a flat numpy array   1x total number of valid pixels in all windows
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)

        # Extract left and right line pixel positions
        leftx = self.nonzerox[self.left_lane_inds]
        lefty = self.nonzeroy[self.left_lane_inds]
        rightx = self.nonzerox[self.right_lane_inds]
        righty = self.nonzeroy[self.right_lane_inds]

        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)

    def visualize(self):
        # Generate x and y values for plotting

        ploty = np.linspace(0, self.img_h - 1, self.img_h)
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]
        self.out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        self.out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]
        plt.imshow(self.out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.show()


        # plotx = np.linspace(0, self.img_w - 1, self.img_w)
        # # x = a * y ^2 + b * y + c
        # left_fity = self.left_fit[0] * plotx ** 2 + self.left_fit[1] * plotx + self.left_fit[2]
        # right_fity = self.right_fit[0] * plotx ** 2 + self.right_fit[1] * plotx + self.right_fit[2]
        # # self.left_fit_poly = np.poly1d(self.left_fit)
        #
        # self.out_img[self.nonzeroy[self.left_lane_inds], self.nonzerox[self.left_lane_inds]] = [255, 0, 0]
        # self.out_img[self.nonzeroy[self.right_lane_inds], self.nonzerox[self.right_lane_inds]] = [0, 0, 255]
        # # plt.plot(plotx, self.left_fit_poly(plotx), color='yellow')
        # plt.plot(plotx, left_fity,  color='yellow')
        # plt.plot(plotx, right_fity, color='yellow')
        # plt.xlim(0, 1280)
        # plt.ylim(720, 0)
        # plt.show()


def main():
    fname = "../test_images/birdeye.jpg"
    outdir = "../output_images/histogram_lane/"
    img = cv2.imread(fname)
    binary_warped = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boundaryTool = Boundary()
    boundaryTool.histogram_peaks(outdir, binary_warped)
    boundaryTool.slid_window()
    boundaryTool.visualize()

if __name__ == "__main__":
    main()
