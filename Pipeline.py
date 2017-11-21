import numpy as np
from MyHelper import abs_sobel_thresh, mag_thresh, combined_result, corners_unwarp, correct_distortion, \
    find_line, back_to_original, base_points, find_line_with_last_data, center_offset_meters
import Lines
import pickle


class Pipeline:
    def __init__(self):
        self.left_line = Lines.Line()
        self.right_line = Lines.Line()
        self.mtx = None
        self.dist = None
        # Get the warped image from the source and destination points of the transformation
        # Moreover, it gets the un warp index get back to the original image
        self.src = np.float32([[280, 670], [565, 470], [720, 470], [1035, 670]])
        self.dst = np.float32([[270, 710], [270, 50], [1035, 50], [1035, 710]])

    def get_pickle_data(self, directory="camera_cal/", file_name="wide_dist_pickle.p"):
        # Read in the saved camera matrix and distortion coefficients
        # These are the arrays you calculated using cv2.calibrateCamera()
        dist_pickle = pickle.load(open(directory + file_name, "rb"))
        self.mtx = dist_pickle["mtx"]
        self.dist = dist_pickle["dist"]

    def pipeline(self, image):
        # Run the function
        image = correct_distortion(image, self.mtx, self.dist)

        gradx = abs_sobel_thresh(image, orient='x', threshold=(15, 255), channel=2)
        grady = abs_sobel_thresh(image, orient='y', threshold=(15, 255), channel=2)
        mag_binary = mag_thresh(image, sobel_kernel=3, threshold=(30, 255), channel=2)
        combined = combined_result(gradx, grady, mag_binary)
        warped, m_inv = corners_unwarp(combined, self.src, self.dst)
        # Find de lines
        # First check if the lines were not find bedore
        if not self.left_line.detected or not self.right_line.detected:
            left_base, right_base = base_points(warped)
            self.left_line.line_base = left_base
            self.right_line.line_base = right_base
            if not self.left_line.detected:
                find_line(warped, self.left_line)
            if not self.right_line.detected:
                find_line(warped, self.right_line)
        else:
            find_line_with_last_data(warped, self.left_line)
            find_line_with_last_data(warped, self.right_line)
            center_offset_meters(self.left_line, self.right_line)

        return back_to_original(warped, image, self.left_line, self.right_line, m_inv)
