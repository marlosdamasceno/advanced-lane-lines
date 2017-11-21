import numpy as np
import cv2


def correct_distortion(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)


# Define a function that applies Sobel x or y, then takes an absolute value and applies a threshold.
def abs_sobel_thresh(img, convert_color=True, color_conversion_between=cv2.COLOR_RGB2HLS,
                     channel=0, orient='x', sobel_kernel=3, threshold=(0, 255)):
    # Check the dimension of the image
    if convert_color:
        converted = cv2.cvtColor(img, color_conversion_between)  # Convert to color space
        img_to_process = converted[:, :, channel]  # Get the channel, the default is 0
    else:
        img_to_process = img[:, :, channel]  # Get the channel, the default is 0

    # Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
    abs_sobel = None
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img_to_process, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img_to_process, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Inclusive threshold
    binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1.0
    # Return the binary image
    return binary_output


def mag_thresh(img, convert_color=True, color_conversion_between=cv2.COLOR_RGB2HLS,
               channel=0, sobel_kernel=3, threshold=(0, 255)):
    # Check the dimension of the image
    if convert_color:
        converted = cv2.cvtColor(img, color_conversion_between)  # Convert to color space
        img_to_process = converted[:, :, channel]  # Get the channel, the default is 0
    else:
        img_to_process = img[:, :, channel]  # Get the channel, the default is 0

    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(img_to_process, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img_to_process, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= threshold[0]) & (gradmag <= threshold[1])] = 1
    # Return the binary image
    return binary_output


def combined_result(gradx, grady, mag_binary, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))):
    combined_img = np.zeros_like(gradx)
    combined_img[((gradx == 1) & (grady == 1)) | (mag_binary == 1)] = 1
    combined_img = cv2.morphologyEx(combined_img, cv2.MORPH_CLOSE, kernel)
    return combined_img


def corners_unwarp(img, src, dst):
    # 1) Define 4 source points src = np.float32([[,],[,],[,],[,]])
    # 2) Define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    # 3) Use cv2.getPerspectiveTransform() to get M, the transform matrix
    # 4) Use cv2.warpPerspective() to warp your image to a top-down view
    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, m, img.shape[1:: -1], flags=cv2.INTER_LINEAR)
    return warped, m_inv


def base_points(img):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    return leftx_base, rightx_base


def find_line(img, line, nwindows=9, margin=100, minpix=50):
    # Set height of windows
    window_height = np.int(img.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    x_current = line.line_base
    # Create empty lists to receive left and right lane pixel indices
    lane_inds = []
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img.shape[0] - (window + 1) * window_height
        win_y_high = img.shape[0] - window * window_height
        win_x_low = x_current - margin
        win_x_high = x_current + margin
        # Identify the nonzero pixels in x and y within the window
        good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                     (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
        # Append these indices to the lists
        lane_inds.append(good_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_inds) > minpix:
            x_current = np.int(np.mean(nonzerox[good_inds]))

    if not lane_inds:
        line.detected = False
        return False

    line.detected = True
    # Concatenate the arrays of indices
    lane_inds = np.concatenate(lane_inds)
    # Extract left and right line pixel positions
    line.allx = nonzerox[lane_inds]
    line.ally = nonzeroy[lane_inds]

    # Fit a second order polynomial to each
    line.current_poly_fit = np.polyfit(line.ally, line.allx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    line.current_x_fitted = line.current_poly_fit[0] * ploty ** 2 + line.current_poly_fit[1] * ploty + \
                            line.current_poly_fit[2]
    line.plot_y = ploty
    line.recent_x_fitted.append(line.current_x_fitted)
    line.recent_poly_fit.append(line.current_poly_fit)
    return True


def find_line_with_last_data(img, line, margin=100):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    if any(line.best_poly_fit):
        lane_inds = ((nonzerox > (line.best_poly_fit[0] * (nonzeroy ** 2) + line.best_poly_fit[1] * nonzeroy +
                                  line.best_poly_fit[2] - margin))
                     & (nonzerox < (line.best_poly_fit[0] * (nonzeroy ** 2) +
                                    line.best_poly_fit[1] * nonzeroy + line.best_poly_fit[2] + margin)))
    else:
        lane_inds = ((nonzerox > (line.current_poly_fit[0] * (nonzeroy ** 2) + line.current_poly_fit[1] * nonzeroy +
                                  line.current_poly_fit[2] - margin))
                     & (nonzerox < (line.current_poly_fit[0] * (nonzeroy ** 2) +
                                    line.current_poly_fit[1] * nonzeroy + line.current_poly_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    allx = nonzerox[lane_inds]
    ally = nonzeroy[lane_inds]
    if any(allx) and any(ally):
        line.allx = allx
        line.ally = ally
        # Fit a second order polynomial to each
        current_fit = np.polyfit(line.ally, line.allx, 2)
        line_curvature_meters(line)
        if line.current_radius_of_curvature <= line.smallest_curvature:
            if any(line.best_poly_fit):
                line.current_poly_fit = line.best_poly_fit
            else:
                line.current_poly_fit = current_fit
        else:
            line.current_poly_fit = current_fit
    else:
        if any(line.best_poly_fit):
            line.current_poly_fit = line.best_poly_fit

    # Generate x and y values for plotting
    line.plot_y = np.linspace(0, img.shape[0] - 1, img.shape[0])
    line.current_x_fitted = line.current_poly_fit[0] * line.plot_y ** 2 + line.current_poly_fit[1] * line.plot_y + \
                            line.current_poly_fit[2]
    if len(line.recent_x_fitted) < line.number_to_average:
        line.recent_x_fitted.append(line.current_x_fitted)
        line.recent_poly_fit.append(line.current_poly_fit)
    else:
        line.recent_x_fitted.pop(0)
        line.recent_x_fitted.append(line.current_x_fitted)
        line.recent_poly_fit.pop(0)
        line.recent_poly_fit.append(line.current_poly_fit)
        line.best_poly_fit = np.mean(line.recent_poly_fit, axis=0)


def line_curvature_meters(line, y_eval=700, ym_per_pix=30 / 720, xm_per_pix=3.7 / 700):
    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(line.ally * ym_per_pix, line.allx * xm_per_pix, 2)
    curvature = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    line.current_radius_of_curvature = curvature
    if curvature > line.smallest_curvature:
        if len(line.recent_radius_of_curvature) < line.number_to_average:
            line.recent_radius_of_curvature.append(line.current_radius_of_curvature)
        else:
            line.recent_radius_of_curvature.pop(0)
            line.recent_radius_of_curvature.append(line.current_radius_of_curvature)
            line.averaged_curvature = np.mean(line.recent_radius_of_curvature)


def center_offset_meters(line_left, line_right, x_mid=640, xm_per_pix=3.7 / 700):
    left_max_y = np.argmax(line_left.plot_y)
    right_max_y = np.argmax(line_right.plot_y)

    left_int = line_left.current_poly_fit[0] * left_max_y ** 2 + line_left.current_poly_fit[1] * left_max_y + \
               line_left.current_poly_fit[2]
    right_int = line_right.current_poly_fit[0] * right_max_y ** 2 + line_right.current_poly_fit[1] * right_max_y + \
                line_right.current_poly_fit[2]

    center = (left_int + right_int) / 2
    center_offset = abs((center - x_mid) * xm_per_pix) * 100 # To get it in centimeters
    line_left.current_center_offset = center_offset
    line_right.current_center_offset = center_offset
    if len(line_left.recent_center_offset) < line_left.number_to_average:
        line_left.recent_center_offset.append(center_offset)
        line_right.recent_center_offset.append(center_offset)
    else:
        line_right.recent_center_offset.pop(0)
        line_right.recent_center_offset.append(center_offset)
        line_right.averaged_center_offset = np.mean(line_right.recent_center_offset)

        line_left.recent_center_offset.pop(0)
        line_left.recent_center_offset.append(center_offset)
        line_left.averaged_center_offset = np.mean(line_left.recent_center_offset)


def back_to_original(warped, image, left_line, right_line, m_inv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_line.current_x_fitted, left_line.plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_line.current_x_fitted, right_line.plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, m_inv, (image.shape[1], image.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    result = cv2.putText(result, "Left radius: " + str(left_line.averaged_curvature) + " m", (20, 40),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    result = cv2.putText(result, "Right radius: " + str(right_line.averaged_curvature) + " m", (20, 80),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    result = cv2.putText(result, "Offset center: " + str(left_line.averaged_center_offset) + " cm", (20, 120),
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    if abs(left_line.current_poly_fit[1]) < 0.06 and abs(right_line.current_poly_fit[1]) < 0.06:
        direction = "Straight"
    else:
        if left_line.current_poly_fit[1] > 0.06 and right_line.current_poly_fit[1] > 0.06:
            direction = "Turning left"
        else:
            if left_line.current_poly_fit[1] < -0.06 and right_line.current_poly_fit[1] < -0.06:
                direction = "Turning right"
            else:
                direction = "Straight"

    result = cv2.putText(result, direction, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    return result
