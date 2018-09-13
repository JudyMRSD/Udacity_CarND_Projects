import cv2
import numpy as np
import matplotlib.pyplot as plt

class Perspective():
    def __init__(self):
        self.a = 0

    # the transformation only needs to be calculated once, assuming the extrinsics and intrinxics did not change
    def warp_front_to_birdeye(self, a1, b1, a2, b2, front_img, out_dir=None):
        h, w = front_img.shape[0:2]
        src = np.float32([[w, h],
                          [0, h],
                          [a1, b1],
                          [a2, b2]])
        dst = np.float32([[w, h],
                          [0, h],
                          [0, 0],
                          [w, 0]])

        to_bird_matrix = cv2.getPerspectiveTransform(src, dst)
        to_front_matrix = cv2.getPerspectiveTransform(dst, src)
        birdeye_img = cv2.warpPerspective(front_img, to_bird_matrix, (w, h))
        cv2.imwrite(out_dir+"birdeye.jpg", birdeye_img)
        if out_dir:
            front_pts = src.reshape(-1, 1, 2).astype(int)
            front_lines = cv2.polylines(front_img, [front_pts], color=(0, 255, 0), thickness=5, isClosed=True)
            cv2.imwrite(out_dir+ "lines_front.jpg", front_lines)
            birdeye_pts = dst.reshape(-1, 1, 2).astype(int)
            birdeye_lines = cv2.polylines(birdeye_img, [birdeye_pts], color=(0, 255, 0), thickness=5, isClosed=True)
            cv2.imwrite(out_dir+ "lines_birdeye.jpg", birdeye_lines)
        return birdeye_img, to_bird_matrix, to_front_matrix

def main():
    car_input_test = '../test_images/straight_lines2.jpg'
    out_dir = "../output_images/birdeye/"
    front_img = cv2.imread(car_input_test)
    perspective_tool = Perspective()
    a1, b1, a2, b2 = 550, 460, 730, 460

    birdeye_img, to_bird_matrix, to_front_matrix = perspective_tool.warp_front_to_birdeye(a1, b1, a2, b2,
                                                                                          front_img, out_dir)


if __name__ == "__main__":
    main()