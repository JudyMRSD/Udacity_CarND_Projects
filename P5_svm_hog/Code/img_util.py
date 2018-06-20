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

    def add_heat(self, heatmap, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    def heat_map(self, image, box_list, writeup_imgs_dir):
        # input: image with bbox
        # output: heatmap
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        # Add heat to each box in box list
        heat = self.add_heat(heat, box_list)
        plt.imshow(heat, cmap='hot')
        plt.savefig(writeup_imgs_dir + "heatmap_original.jpg")
        plt.clf()
        # Apply threshold to help remove false positives
        threshold = 20
        heat[heat <= threshold] = 0
        plt.imshow(heat, cmap='hot')
        plt.colorbar()
        plt.savefig(writeup_imgs_dir + "heatmap_thresh.jpg")
        plt.clf()

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)
        # Find final boxes from heatmap using label function
        # Create an image with some features, then label it using the default (cross-shaped) structuring element
        # (if 2 pixels are connected in vertical or horizontal orientation, they are same object)
        labels = label(heatmap)
        draw_img = self.draw_labeled_bboxes(np.copy(image), labels)

        cv2.imwrite(writeup_imgs_dir + "bbox_heatmap.jpg", draw_img)


    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        # Return the image
        return img


