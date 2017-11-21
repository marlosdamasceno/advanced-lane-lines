import numpy as np


# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self):
        self.number_to_average = 10
        self.smallest_curvature = 600

        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # was the line detected in the last iteration?
        self.detected = False
        # base position of line
        self.line_base = None
        # poly fits, polynomial coefficients for the most recent fit
        self.current_poly_fit = [np.array([False])]
        # poly fits, polynomial coefficients for the most recent fit
        self.recent_poly_fit = []
        # polynomial coefficients averaged over the last n iterations
        self.best_poly_fit = []
        # current x fitted
        self.current_x_fitted = [np.array([False])]
        # x values of the last n fits of the line
        self.recent_x_fitted = []
        # y values of pixeis
        self.plot_y = [np.array([False])]
        # radius of curvature of the line in some units
        self.current_radius_of_curvature = None
        # past values of curvature
        self.recent_radius_of_curvature = []
        # averaged curvature
        self.averaged_curvature = None
        # distance in meters of vehicle center from the line
        self.current_center_offset = None
        # past values of offset
        self.recent_center_offset = []
        # averaged offset
        self.averaged_center_offset = None
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
