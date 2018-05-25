import cv2
# Udacity 18 :  Detect lane pixels and fit to find the lane boundary
class Boundary():
    def __init__(self):
        self.a  = 0
    def find_peaks(self, img):

        pass

def main():
    fname = "../test_images/birdeye_straight_lines2.jpg"
    img = cv2.imread(fname)

if __name__ == "__main__":
    main()
