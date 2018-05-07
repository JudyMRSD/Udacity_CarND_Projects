import cv2
import numpy as np
class ImagePreprocess:
    def __init__(self):
        self.a = 0
    def binary_HSV(self, input_img):
        img = cv2.imread(input_img)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def binary_HLS(self, input_img):
        '''
        S: saturation picks up the lines well
        :param input_img:
        :return:
        '''
        print("binary HLS")
        img = cv2.imread(input_img)
        print("img", img.shape)
        hls_img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        H = hls_img[:,:,0]
        L = hls_img[:,:,1]
        S = hls_img[:,:,2]

        thresh = 90
        maxval = 255
        # dst(x,y) = maxval if src(x,y) > thresh,   dst(x,y)=0 otherwise
        ret, H_binary = cv2.threshold(S, thresh, maxval, cv2.THRESH_BINARY)
        cv2.imwrite("../output_images/thresh_out/H.jpg", H)
        cv2.imwrite("../output_images/thresh_out/L.jpg", L)
        cv2.imwrite("../output_images/thresh_out/S.jpg", S)
        cv2.imwrite("../output_images/thresh_out/S_binary.jpg", H_binary)

    def binaryLane(self):
        '''
        create a binary image where lanes are white pixels
        :return:
        '''
        pass

def main():
    imgProcessor = ImagePreprocess()
    input_img = '../output_images/test_image_out/test2_undist.jpg'

    print("binary")
    imgProcessor.binary_HSV(input_img)
    imgProcessor.binary_HLS(input_img)

if __name__ == "__main__":
    main()