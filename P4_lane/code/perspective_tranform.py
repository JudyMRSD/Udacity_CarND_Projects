import cv2
import numpy as np
import matplotlib.pyplot as plt


class Perspective():
    def __init__(self):
        self.a = 0
    def warp_front_to_birdeye(self, front_img, verbose):
        h, w, _ = front_img.shape
        # todo: modify src dst points   input xmin=546, xmax=732, ymin=0, ymax=460
        src = np.float32([[w, h - 10],  # br
                          [0, h - 10],  # bl
                          [546, 460],  # tl
                          [732, 460]])  # tr
        dst = np.float32([[w, h],  # br
                          [0, h],  # bl
                          [0, 0],  # tl
                          [w, 0]])  # tr

        to_bird_matrix = cv2.getPerspectiveTransform(src, dst)
        to_front_matrix = cv2.getPerspectiveTransform(dst, src)
        birdeye_img = cv2.warpPerspective(front_img, to_bird_matrix, (w, h))
        if verbose is True:
            src = src.reshape(-1, 1, 2).astype(int)
            img_lines = cv2.polylines(front_img, src, color=(0, 255, 0), thickness=20, isClosed=True)
            cv2.imwrite("../output_images/lines_warped.jpg", img_lines)
        return birdeye_img, to_bird_matrix, to_front_matrix

def main():
    car_input_test = '../test_images/test2.jpg'
    front_img = cv2.imread(car_input_test)
    perspective_tool = Perspective()
    birdeye_img, to_bird_matrix, to_front_matrix = perspective_tool.warp_front_to_birdeye(front_img, verbose = True)
    cv2.imwrite("../output_images/birdeye_img.jpg",birdeye_img)

    # plot the 4 lines on original image


if __name__ == "__main__":
    main()