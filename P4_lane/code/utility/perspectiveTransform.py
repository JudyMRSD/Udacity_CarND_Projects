import cv2
import numpy as np

class Perspective():
    def __init__(self):
        self.a = 0

    # the transformation only needs to be calculated once, assuming the extrinsics and intrinxics did not change
    def warp_front_to_birdeye(self, lane_ends, front_img, out_dir=None):
        left_lane_top_x, left_lane_top_y, right_lane_top_x, right_lane_top_y = lane_ends
        h, w = front_img.shape[0:2]
        src = np.float32([[w, h],
                          [0, h],
                          [left_lane_top_x, left_lane_top_y],
                          [right_lane_top_x, right_lane_top_y]])
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
            print("front_pts", front_pts)
            front_lines = cv2.polylines(front_img, [front_pts], color=(0, 255, 0), thickness=5, isClosed=True)
            cv2.imwrite(out_dir+ "lines_front.jpg", front_lines)
            birdeye_pts = dst.reshape(-1, 1, 2).astype(int)
            birdeye_lines = cv2.polylines(birdeye_img, [birdeye_pts], color=(0, 255, 0), thickness=5, isClosed=True)
            cv2.imwrite(out_dir+ "lines_birdeye.jpg", birdeye_lines)
        return birdeye_img, to_bird_matrix, to_front_matrix
