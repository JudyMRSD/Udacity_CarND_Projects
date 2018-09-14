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
            print("out_dir", out_dir)
            front_color = cv2.cvtColor(front_img, cv2.COLOR_GRAY2RGB)
            cv2.circle(front_color, (200, 200), 55, (0, 255, 255), -1)
            print("front_img.shape", front_color.shape)
            cv2.imwrite(out_dir + "lines_front.jpg", front_img)
            cv2.imshow("front", front_color)
            cv2.waitKey(0)
            src = np.float32([[h, w],
                              [h, 0],
                              [left_lane_top_y, left_lane_top_x],
                              [right_lane_top_y, right_lane_top_x]])
            front_pts = src.reshape(-1, 1, 2).astype(np.int32)
            print("front_pts", front_pts)
            front_lines = cv2.polylines(front_color
                                        , front_pts, color=(0, 255, 0), thickness=5, isClosed=False)
            cv2.imwrite(out_dir + "lines_front.jpg", front_lines)

            birdeye_pts = dst.reshape(-1, 1, 2).astype(np.int32)
            birdeye_lines = cv2.polylines(birdeye_img, birdeye_pts, color=(0, 255, 0), thickness=5, isClosed=False)
            cv2.imwrite(out_dir+ "lines_birdeye.jpg", birdeye_lines)
        return birdeye_img, to_bird_matrix, to_front_matrix
