import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt


# todo: read image use opencv, not mpimg
class ImgUtil:
    def __init__(self):
        self.a = 0

    def convert_color(self, image, color_space):
        # opencv read image input as BGR
        if color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif color_space == 'RGB':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            print("Error: invalid color_space")

        return feature_image




