import cv2
import matplotlib.pyplot as plt
import numpy as np
from utility.perspectiveTransform import Perspective

# Used code from Udacity online course 18 :  Detect lane pixels and fit to find the lane boundary
class Boundary():
    def __init__(self):
        self.margin = 100
        # Choose the number of sliding windows
        self.nwindows = 9
        # Set the width of the windows +/- margin
        self.margin = 100
        # Set minimum number of pixels found to recenter window
        self.minpix = 50

    ####################################
    # Helper functions
    ####################################
    def y_to_x(self, y):
        # calculate x = ay**2 + by + c
        # self.left_fit and self.right_fit contains [a,b,c]
        print("self.left_fit" , self.left_fit)
        self.left_fit_x = self.left_fit[0] * (y ** 2) + self.left_fit[1] * y + self.left_fit[2]
        self.right_fit_x = self.right_fit[0] * (y ** 2) + self.right_fit[1] * y + self.right_fit[2]
        print("self.left_fit_x.shape", self.left_fit_x.shape,"y.shape", y.shape )

    @staticmethod
    def polynomial_to_points(left_x, right_x, y):
        print("left_x, y", left_x.shape, y.shape)
        left_vertices = np.array([np.transpose(np.vstack([left_x, y]))], dtype=np.int32)
        # flip the points on the right edge of the left traffic lane, so the points are ordered for fillPoly
        # 1               6
        # 2               5
        # 3               4
        # left window    right window
        right_vertices = np.array([np.flipud(np.transpose(np.vstack([right_x, y])))], dtype=np.int32)
        pts = np.hstack((left_vertices, right_vertices))

        return pts, left_vertices, right_vertices


    def histogram_peaks(self, outdir, img):
        self.img = img
        assert (len(np.unique(self.img) == 2) , "input to histogram_peaks must be binary image, with values 0 or 255")
        # take a histogram along all the columns in the lower half of the image
        self.img_h, self.img_w = self.img.shape[0:2]
        histogram = np.sum(self.img[self.img_h // 2:, :], axis=0)
        plt.plot(histogram)
        plt.xlabel('Pixel Positions')
        plt.ylabel('Counts')

        if (outdir):
            plt.savefig(outdir + "histogram.jpg")
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = int(histogram.shape[0] // 2)
        self.leftx_base = np.argmax(histogram[:midpoint])
        self.rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        print("self.leftx_base", self.leftx_base,"self.rightx_base", self.rightx_base )
        plt.close()

    def calc_curvature(self):
        y = self.img_h - 10
        # value taken from Udacity course
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(self.lefty * ym_per_pix, self.leftx * self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.righty * ym_per_pix, self.rightx * self.xm_per_pix, 2)
        # Calculate the new radii of curvature a particular y coordinate
        left_curverad = ((1 + (2 * left_fit_cr[0] * y * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
        # Now our radius of curvature is in meters
        return left_curverad, right_curverad

    def calc_dist_center(self):
        # assume center of image is along the center line of vehicle
        center_car_x = int(self.img_w / 2)
        # only take the x value for the left and right line polynomial towards bottom of frame
        center_lane_x = int((self.left_fit_x[-1]  + self.right_fit_x[-1])/2)
        dist = (center_lane_x - center_car_x) * self.xm_per_pix
        return dist

    def create_window(self, i):
        # Set height of windows (image height divided by number of sliding windows)
        window_height = np.int(self.img_h // self.nwindows)

        # Identify window boundaries in y
        self.win_y_low = self.img_h - (i + 1) * window_height
        self.win_y_high = self.img_h - i * window_height
        # Identify window boundaries in x for left lane
        self.win_xleft_low = self.leftx_current - self.margin
        self.win_xleft_high = self.leftx_current + self.margin
        # Identify window boundaries in x for right lane
        self.win_xright_low = self.rightx_current - self.margin
        self.win_xright_high = self.rightx_current + self.margin
        # visualize location of current windows
        cv2.rectangle(self.out_img, (self.win_xleft_low, self.win_y_low), (self.win_xleft_high, self.win_y_high),
                      (0, 255, 0), 2)
        cv2.rectangle(self.out_img, (self.win_xright_low, self.win_y_low), (self.win_xright_high, self.win_y_high),
                      (0, 255, 0), 2)

    def recenter_window(self):
        # Part 1: find indices for pixels within the current window that have nonzero intensity
        # bool_in_window_left[i] is true when self.nonzeroy[i] and self.nonzerox[i] is a point in window
        bool_in_window_left = ((self.nonzeroy >= self.win_y_low) & (self.nonzeroy < self.win_y_high) &
                          (self.nonzerox >= self.win_xleft_low) & (self.nonzerox < self.win_xleft_high))
        # convert from [False True ....] to [1 ...]
        idx_in_window_left = np.nonzero(bool_in_window_left)
        # array of one list, take 0th list to reduce the dimension
        idx_in_window_left = idx_in_window_left[0]
        # Append these indices to the lists
        self.left_lane_inds.append(idx_in_window_left)

        idx_in_window_right = ((self.nonzeroy >= self.win_y_low) & (self.nonzeroy < self.win_y_high) &
                           (self.nonzerox >= self.win_xright_low) & (self.nonzerox < self.win_xright_high)).nonzero()[0]
        self.right_lane_inds.append(idx_in_window_right)

        # Part 2: recenter window for left or right lane

        # If you found > minpix pixels, recenter next window on their mean position
        if len(idx_in_window_left) > self.minpix:
            # x coordinates for pixels that has nonzero intensity (belong to lane) and also lies inside current window
            left_nonzero_in_window_x_coords = self.nonzerox[idx_in_window_left]
            # recenter next window on their mean position
            self.leftx_current = np.int(np.mean(left_nonzero_in_window_x_coords))

        if len(idx_in_window_right) > self.minpix:
            right_nonzero_in_window_x_coords = self.nonzerox[idx_in_window_right]
            self.rightx_current = np.int(np.mean(right_nonzero_in_window_x_coords))


    def slide_window(self):
        # Current positions to be updated for each window
        self.leftx_current = self.leftx_base
        self.rightx_current = self.rightx_base

        # Step through the windows one by one
        # Draw the windows on the visualization image
        self.out_img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2BGR)
        for i in range(self.nwindows):
            self.create_window(i)
            self.recenter_window()

        # Concatenate the arrays of indices:
        # left_lane_inds  num window rows, each row has num valid pixels in the window
        # turn this into a flat numpy array   1x total number of valid pixels in all windows
        self.left_lane_inds = np.concatenate(self.left_lane_inds)
        self.right_lane_inds = np.concatenate(self.right_lane_inds)




    def fit_use_prev(self):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # You don't need to do a blind search again, but instead you can just search
        # in a margin around the previous line position l
        self.y_to_x(self.nonzeroy)

        self.left_lane_inds = ((self.nonzerox > self.left_fit_x  - self.margin)
                          & (self.nonzerox < self.left_fit_x  + self.margin))

        self.right_lane_inds = ((self.nonzerox > (self.right_fit_x  - self.margin))
                           & (self.nonzerox < self.right_fit_x  + self.margin))

    ####################################
    # fit_lane and visualize_lane are called by pipeline.py
    ####################################
    # this calls slide window or fit using previous result depends
    def fit_lane(self, birdeye_binary, idx):
        # Identify the x and y positions of all nonzero pixels in the image
        self.img = birdeye_binary
        nonzero_pix = self.img.nonzero()
        self.nonzeroy = np.array(nonzero_pix[0])  # y coordinate of non zero pixel (row)
        self.nonzerox = np.array(nonzero_pix[1])  # x coordinate of non zero pixel  (col)

        # Create empty lists to receive left and right lane pixel indices
        self.left_lane_inds = []
        self.right_lane_inds = []

        if idx==0:
            self.slide_window()
        else:
            self.fit_use_prev()

        # Extract left and right line pixel positions
        self.leftx = self.nonzerox[self.left_lane_inds]
        self.lefty = self.nonzeroy[self.left_lane_inds]
        self.rightx = self.nonzerox[self.right_lane_inds]
        self.righty = self.nonzeroy[self.right_lane_inds]

        # Fit a second order polynomial to each
        # use function x(y) instead of y(x)
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)




    def visualize_lane(self, outdir, original_img, to_front_matrix, blend_alpha = 0.5):

        # Part 1: Create an image with lanes in bird-eye view, then warp it to front view
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, original_img.shape[0] - 1, original_img.shape[0])
        self.y_to_x(self.ploty)

        # input: 3 channel warped image
        color_birdeye_mask = np.zeros_like(original_img)
        pts, left_vertices, right_vertices = self.polynomial_to_points(self.left_fit_x, self.right_fit_x, self.ploty)
        # draw lane boundaries on the warped imagely
        cv2.fillPoly(color_birdeye_mask, [pts], (0,255,0))
        cv2.polylines(color_birdeye_mask,[left_vertices], isClosed = False, color =  (255, 0,0), thickness = 20)
        cv2.polylines(color_birdeye_mask, [right_vertices], isClosed = False, color =  (0, 0, 255), thickness = 20)
        # warp the mask back to original image (front view)
        color_front_mask = cv2.warpPerspective(color_birdeye_mask,to_front_matrix , (self.img_w, self.img_h))

        # Part 2: blend original image with the lanes
        img_blend = np.zeros_like(original_img)
        blend_beta = 1- blend_alpha
        result_img = cv2.addWeighted(color_front_mask, blend_alpha, original_img, blend_beta, 0.0, img_blend)

        # Part 3: show curvature and distance to center

        # calculate curvature and center dist
        left_curverad, right_curverad = self.calc_curvature()
        center_dist = self.calc_dist_center()

        # put text on image
        curv_left_text =  "Left  lane curvature radius = {0:.2f} m".format(left_curverad)
        curv_right_text = "Right lane curvature radius = {0:.2f} m".format(left_curverad)
        center_dist_text ="Distance to center = {0:.2f} m".format(center_dist)

        cv2.putText(result_img, curv_left_text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result_img, curv_right_text, (10, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(result_img, center_dist_text, (10, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if(outdir):
            cv2.imwrite(outdir + "lane_border_birdeye.jpg", color_birdeye_mask)
            cv2.imwrite(outdir + "lane_border_front.jpg", color_front_mask)
            cv2.imwrite(outdir + "blend.jpg", result_img)

        return result_img
