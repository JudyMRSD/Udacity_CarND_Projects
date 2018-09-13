import cv2
import matplotlib.pyplot as plt
import numpy as np
from perspective_transform import Perspective

# Used code from Udacity online course 18 :  Detect lane pixels and fit to find the lane boundary
class Boundary():
    def __init__(self):
        self.a  = 0
    def histogram_peaks(self, outdir, img):
        self.img = img
        assert (len(np.unique(self.img) == 2) , "input to histogram_peaks must be binary image, with values 0 or 255")
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

    def slide_window(self):
        self.img_h, self.img_w = self.img.shape[0:2]
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows (image height divided by number of sliding windows)
        window_height = np.int(self.img_h // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero_pix = self.img.nonzero()
        self.nonzeroy = np.array(nonzero_pix[0]) # y coordinate of non zero pixel (row)
        self.nonzerox = np.array(nonzero_pix[1]) # x coordinate of non zero pixel  (col)
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
                              (self.nonzerox >= win_xleft_low) & (self.nonzerox < win_xleft_high))
            good_left_inds = np.nonzero(good_left_inds)
            good_left_inds = good_left_inds[0]
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
        self.leftx = self.nonzerox[self.left_lane_inds]
        self.lefty = self.nonzeroy[self.left_lane_inds]
        self.rightx = self.nonzerox[self.right_lane_inds]
        self.righty = self.nonzeroy[self.right_lane_inds]

        # Fit a second order polynomial to each
        # use function x(y) instead of y(x) since polyfit needs input x, y in increasing order
        # treat x as vertical axis and y as horizontal axis ensures both vertical and horizontal axis are in increasing order
        # see writeup images
        self.left_fit = np.polyfit(self.lefty, self.leftx, 2)
        self.right_fit = np.polyfit(self.righty, self.rightx, 2)
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, self.img.shape[0] - 1, self.img.shape[0])

        self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]
        self.margin = 100

    def fit_use_prev(self, binary_warped):
        # Assume you now have a new warped binary image
        # from the next frame of video (also called "binary_warped")
        # You don't need to do a blind search again, but instead you can just search
        # in a margin around the previous line position l
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        self.margin = 100

        left_x = self.left_fit[0] * (nonzeroy ** 2) + self.left_fit[1] * nonzeroy + self.left_fit[2]
        right_x = self.right_fit[0] * (nonzeroy ** 2) + self.right_fit[1] * nonzeroy + self.right_fit[2]
        self.left_lane_inds = ((nonzerox > left_x - self.margin) & (nonzerox < left_x + self.margin))
        right_lane_inds = ((nonzerox > (right_x - self.margin)) & (nonzerox < right_x + self.margin))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[self.left_lane_inds]
        lefty = nonzeroy[self.left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        self.left_fit = np.polyfit(lefty, leftx, 2)
        self.right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        self.ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        self.left_fitx = self.left_fit[0] * self.ploty ** 2 + self.left_fit[1] * self.ploty + self.left_fit[2]
        self.right_fitx = self.right_fit[0] * self.ploty ** 2 + self.right_fit[1] * self.ploty + self.right_fit[2]


    @staticmethod
    def polynomial_to_points(x, y, margin):
        left_vertices = np.array([np.transpose(np.vstack([x - margin, y]))]).astype(int)
        # flip the points on the right edge of the left traffic lane, so the points are ordered for fillPoly
        # 1              6
        # 2              5
        # 3              4
        # window    window 2
        right_vertices = np.array([np.flipud(np.transpose(np.vstack([x + margin, y])))]).astype(int)
        pts = np.hstack((left_vertices, right_vertices)).astype(int)
        return pts,left_vertices, right_vertices


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
        print(left_curverad, 'm', right_curverad, 'm')
        return left_curverad, right_curverad

    def calc_dist_center(self):
        # assume center of image is along the center line of vehicle
        center_car_x = int(self.img_w / 2)
        # only take the x value for the left and right line polynomial towards bottom of frame
        center_lane_x = int((self.left_fitx[-1]  + self.right_fitx[-1])/2)
        dist = (center_lane_x - center_car_x) * self.xm_per_pix
        print("dist ", dist)
        return dist

    def visualize_lane(self, outdir, original_img, to_front_matrix, blend_alpha = 0.5):
        # input: 3 channel warped image
        color_birdeye_mask = np.zeros_like(original_img)
        pts, left_vertices, right_vertices = self.polynomial_to_points(self.left_fitx, self.right_fitx, self.margin)
        # draw lane boundaries on the warped imagely
        # todo: color_birdeye_mask is still zeros after fillPoly
        cv2.fillPoly(color_birdeye_mask, [pts], (0,255,0))
        cv2.polylines(color_birdeye_mask, [left_vertices], isClosed = False, color =  (255, 0,0), thickness = 10)
        cv2.polylines(color_birdeye_mask, [right_vertices], isClosed = False, color =  (0, 0, 255), thickness = 10)
        # Idea taken from https://github.com/jeremy-shannon/CarND-Advanced-Lane-Lines/blob/master/project.ipynb
        # warp the mask back to original image (fornt view)
        color_front_mask = cv2.warpPerspective(color_birdeye_mask,to_front_matrix , (self.img_w, self.img_h))
        # blend
        img_blend = np.zeros_like(original_img)
        blend_beta = 1- blend_alpha
        result_img = cv2.addWeighted(color_front_mask, blend_alpha, original_img, blend_beta, 0.0, img_blend)

        left_curverad, right_curverad = self.calc_curvature()
        center_dist = self.calc_dist_center()

        text = "Left curvature = {0:.3f} m \n Right curvature = {0:.3f} m \n Distance to center = center_dist \n"\
            .format(left_curverad, right_curverad, center_dist)

        cv2.putText(result_img, text, (40, 120), cv2.FONT_HERSHEY_DUPLEX, 1.5, (200, 255, 155), 2, cv2.LINE_AA)
        if(outdir):
            cv2.imwrite(outdir + "blend.jpg", result_img)

        return result_img



def main():
    to_front_matrix = np.array([[ 1.40625000e-01, -7.63888889e-01,  5.50000000e+02],
                                 [ 1.19904087e-14, -4.98263889e-01,  4.60000000e+02],
                                 [ 1.99493200e-17, -1.19357639e-03,  1.00000000e+00]]).astype(int)

    birdeye_binary_img = "../test_images/birdeye.jpg"
    outdir = "../output_images/histogram_lane/"
    warped_img = cv2.imread(birdeye_binary_img)
    binary_warped = cv2.cvtColor(warped_img, cv2.COLOR_BGR2GRAY)
    boundaryTool = Boundary()
    boundaryTool.histogram_peaks(outdir, binary_warped)
    boundaryTool.slide_window(outdir)
    boundaryTool.fit_use_prev(binary_warped, outdir)
    boundaryTool.calc_curvature()
    boundaryTool.calc_dist_center()

    front_img = cv2.imread('../test_images/straight_lines2.jpg')
    boundaryTool.visualize_lane(outdir, front_img, to_front_matrix)

if __name__ == "__main__":
    main()