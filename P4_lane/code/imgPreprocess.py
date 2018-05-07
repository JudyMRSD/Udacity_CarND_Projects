import cv2
import numpy as np
import matplotlib.pyplot as plt


out_dir = "../output_images/thresh_out/"
class ImagePreprocess:
    def __init__(self):
        self.a = 0
    def binary_HSV(self, input_img):
        img = cv2.imread(input_img)
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # example: binary (color_space_name = 'HLS', thresh_channel_names = ['s'], input_img = imgpath, thresh = [(90, 255)])
    def binary(self, color_space_name, thresh_channel_names, input_img, thresh):
        img = cv2.imread(input_img)
        if (color_space_name == "HLS"):
            channel_names = ['H', 'L','S']
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
            #thresh = (70, 255)
        elif (color_space_name == "HSV"):
            channel_names = ['H', 'S', 'V']
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HVS)
            #thresh = (90, 255)
        elif (color_space_name == "RGB"):
            channel_names = ['R','G', 'B']
            #thresh = (200, 255)
        elif (color_space_name == "gray"):
            channel_names = ['gray']
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # thresh = (180, 255)
        num_channels = img.shape[2]
        plt.close('all')
        fig, ax = plt.subplots(2,3)
        for i in range(0, num_channels):
            channel = img[:,:,i]
            ax[0][i].imshow(channel, cmap = 'gray')
            ax[0][i].set_title(color_space_name+", channel = " + channel_names[i])
            print("channel_names[i]",channel_names[i],  thresh_channel_names)
            if channel_names[i] in thresh_channel_names:
                print(" channel_names[i] ",  channel_names[i])
                ret, channel_binary = cv2.threshold(channel, thresh[0][0], thresh[0][1], cv2.THRESH_BINARY)
                ax[1][i].imshow(channel_binary, cmap = 'gray')
                ax[1][i].set_title(color_space_name + ", binary thresh = " + channel_names[i])
        plt.savefig(out_dir + channel_names[i] + ".jpg")





    def binary_HLS(self, input_img):
        '''
        S: lines appear white,
        H: lines appear dark
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
    #imgProcessor.binary_HSV(input_img)
    #imgProcessor.binary_HLS(input_img)
    imgProcessor.binary(color_space_name='RGB', thresh_channel_names=['R'], input_img=input_img, thresh=[(200, 255)])

if __name__ == "__main__":
    main()