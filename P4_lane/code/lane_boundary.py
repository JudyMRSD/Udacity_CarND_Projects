import cv2
import matplotlib.pyplot as plt
import numpy as np


# Udacity 18 :  Detect lane pixels and fit to find the lane boundary
class Boundary():
    def __init__(self):
        self.a  = 0
    def histogram_peaks(self, outdir, img):
        print("img shape", img.shape)
        #img = img/255
        # take a histogram along all the columns in the lower half of the image
        histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
        print("histogram shape", histogram.shape)
        plt.plot(histogram)
        plt.xlabel('Counts')
        plt.ylabel('Pixel Positions')
        plt.savefig(outdir + "channels.jpg")
        out_img = np.dstack((img, img, img))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        print(histogram.shape[0],midpoint,leftx_base, rightx_base)


    def slid_window(self):
        pass



def main():
    fname = "../test_images/birdeye.jpg"
    outdir = "../output_images/histogram_lane/"
    img = cv2.imread(fname)
    binary_front_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    boundaryTool = Boundary()
    boundaryTool.histogram_peaks(outdir, binary_front_img)


if __name__ == "__main__":
    main()
