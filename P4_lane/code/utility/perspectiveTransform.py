# Perform perspective transformation to warp front view to bird-eye view
import cv2
import numpy as np

# Points (hand selected) needed to warp images to bird eye view, these are hand selected points
# lane_ends = [left_lane_top_x, left_lane_top_y, right_lane_top_x, right_lane_top_y] encodes the end of two lanes
# that maps to (0,0) and (w,0) in bird eye view, doesn't change if intrinsics and extrinsics not changed
Left_lane_top_x, Left_lane_top_y, Right_lane_top_x, Right_lane_top_y = 550, 460, 730, 460

class Perspective():
    '''
    Calculate the transformation matrix between birdeye view and front view
    the transformation only needs to be calculated once,
    assuming the extrinsics and intrinxics did not change
    '''
    def warp_front_to_birdeye(self, front_img, out_dir=None):
        h, w = front_img.shape[0:2]
        src = np.float32([[w, h],
                          [0, h],
                          [Left_lane_top_x, Left_lane_top_y],
                          [Right_lane_top_x, Right_lane_top_y]])
        dst = np.float32([[w, h],
                          [0, h],
                          [0, 0],
                          [w, 0]])

        to_bird_matrix = cv2.getPerspectiveTransform(src, dst)
        to_front_matrix = cv2.getPerspectiveTransform(dst, src)
        birdeye_img = cv2.warpPerspective(front_img, to_bird_matrix, (w, h))

        if out_dir:
            cv2.imwrite(out_dir + "birdeye.jpg", birdeye_img)

            front_color = cv2.cvtColor(front_img, cv2.COLOR_GRAY2RGB)
            birdeye_color = cv2.cvtColor(birdeye_img, cv2.COLOR_GRAY2RGB)

            front_pts = src.reshape(-1, 1, 2).astype(np.int32)
            front_lines = cv2.polylines(front_color, front_pts, color=(0, 255, 0), thickness=50, isClosed=True)
            cv2.imwrite(out_dir + "lines_front.jpg", front_lines)

            birdeye_pts = dst.reshape(-1, 1, 2).astype(np.int32)
            birdeye_lines = cv2.polylines(birdeye_color, birdeye_pts, color=(0, 255, 0), thickness=50, isClosed=True)
            cv2.imwrite(out_dir+ "lines_birdeye.jpg", birdeye_lines)

        return birdeye_img, to_bird_matrix, to_front_matrix
