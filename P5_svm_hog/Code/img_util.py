import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from scipy.ndimage.measurements import label
import matplotlib.pyplot as plt
plt.switch_backend('agg')


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
        elif color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'GRAY':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            print("Error: invalid color_space")

        return feature_image

    def heatmap_gray_to_rgb(self, gray_heatmap):
        print("gray_heatmap", gray_heatmap.shape)
        rgb_img = cv2.cvtColor(gray_heatmap.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # rgb_img = cv2.applyColorMap(rgb_img, cv2.COLORMAP_JET)

        # cmap = plt.get_cmap('jet')
        # rgba_img = cmap(gray_heatmap)
        # rgb_img = np.delete(rgba_img, 3, 2)# remove 'a' channel for transparency
        return rgb_img

    def add_heat(self, heatmap, bbox_hist):
        # Iterate through list of bboxes
        print("bbox_hist", bbox_hist)
        bbox_hist = np.array(bbox_hist)

        for bbox_list in bbox_hist:
            #print("bbox_list", bbox_list)
            for box in bbox_list:
                #print("box", box)
                # Add += 1 for all pixels inside each bbox
                # Assuming each "box" takes the form ((x1, y1), (x2, y2))
                heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap  # Iterate through list of bboxes

    def heat_map(self, image, bbox_hist, writeup_imgs_dir, threshold, verbose):
        # input: image with bbox
        # output: heatmap
        heat = np.zeros_like(image[:, :, 0]).astype(np.float)
        # Add heat to each box in box list
        print("zeros heatmap", heat.shape)

        heat = self.add_heat(heat, bbox_hist)
        if verbose:
            plt.imshow(heat, cmap='hot')
            plt.savefig(writeup_imgs_dir + "heatmap_original.jpg")
            plt.clf()

        # Apply threshold to help remove false positives
        heat[heat <= threshold] = 0
        if verbose:
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
        draw_heatmap = self.draw_labeled_bboxes(self.heatmap_gray_to_rgb(heatmap), labels)
        return draw_img, draw_heatmap

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
            cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 6)
        # Return the image
        return img
