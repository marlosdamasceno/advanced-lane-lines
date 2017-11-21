import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Button
from matplotlib.patches import Circle
from matplotlib.widgets import TextBox


def corners_unwarp(event):
    # 1) Define 4 source points src = np.float32([[,],[,],[,],[,]])
    # 2) Define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    # 3) Use cv2.getPerspectiveTransform() to get M, the transform matrix
    # 4) Use cv2.warpPerspective() to warp your image to a top-down view
    if len(original_circles) == 4 and len(transform_circles) == 4:
        src = np.float32([original_circles[0].center, original_circles[1].center, original_circles[2].center,
                          original_circles[3].center])
        dst = np.float32([transform_circles[0].center, transform_circles[1].center, transform_circles[2].center,
                          transform_circles[3].center])
        M = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image_transform, M, image_transform.shape[1:: -1], flags=cv2.INTER_LINEAR)
        im2.set_data(warped)
        fig.canvas.draw()
        print(src)
        print(dst)


def reset(event):
    for circle in original_circles:
        circle.remove()
    original_circles.clear()
    im1.set_data(image_original)

    for circle in transform_circles:
        circle.remove()
    transform_circles.clear()
    im2.set_data(image_transform)

    fig.canvas.draw()


directory = 'camera_cal/'  # Directory with the images for calibration
# Read in the saved camera matrix and distortion coefficients
# These are the arrays you calculated using cv2.calibrateCamera()
dist_pickle = pickle.load(open(directory + "wide_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

image_original = mpimg.imread('TestImages/straight_lines1.jpg')
image_original = cv2.undistort(image_original, mtx, dist, None, mtx)
image_transform = np.copy(image_original)

fig = plt.figure()
plt.subplots_adjust(bottom=0.2)
ax_original = fig.add_subplot(1, 2, 1)
im1 = ax_original.imshow(image_original)
ax_transform = fig.add_subplot(1, 2, 2)
im2 = ax_transform.imshow(image_transform)

original_circles = []
transform_circles = []

ax_reset = plt.axes([0.7, 0.05, 0.1, 0.075])
b_reset = Button(ax_reset, 'Reset')
b_reset.on_clicked(reset)

ax_apply = plt.axes([0.81, 0.05, 0.1, 0.075])
b_apply = Button(ax_apply, 'Apply')
b_apply.on_clicked(corners_unwarp)


def submit(text):
    data = str(text)
    source_dest = data.split("|")
    if len(source_dest) == 2:
        source_coord = source_dest[0].split(";")
        if len(source_coord) == 4:
            for coord, index in zip(source_coord, range(4)):
                tuple_coord = (float(coord.split(",")[0]), float(coord.split(",")[1]))
                if len(original_circles) < 4:
                    circ = Circle(tuple_coord, 10, color='red')
                    ax_original.add_patch(circ)
                    original_circles.append(circ)
                else:
                    original_circles[index].center = tuple_coord

        dest_coord = source_dest[1].split(";")
        if len(dest_coord) == 4:
            for coord, index in zip(dest_coord, range(4)):
                tuple_coord = (float(coord.split(",")[0]), float(coord.split(",")[1]))
                if len(transform_circles) < 4:
                    circ = Circle(tuple_coord, 10)
                    ax_transform.add_patch(circ)
                    transform_circles.append(circ)
                else:
                    transform_circles[index].center = tuple_coord

    fig.canvas.draw()


axbox = plt.axes([0.1, 0.1, 0.5, 0.05])
text_box = TextBox(axbox, 'Evaluate',
                   initial="280,670;565,470;720,470;1035,670|270,710;270,50;1035,50;1035,710")
# 280,670;580,460;700,460;1035,670|270,710;270,50;1035,50;1035,710
# 280,670;550,480;735,480;1035,670|270,710;270,50;1035,50;1035,710
text_box.on_submit(submit)


def onclick(event):
    # Check if the click was in ax4 or ax5
    if not event.dblclick:
        if event.button == 1:
            if event.inaxes in [ax_original]:
                if len(original_circles) < 4:
                    circ = Circle((event.xdata, event.ydata), 10, color='red')
                    ax_original.add_patch(circ)
                    original_circles.append(circ)
                    fig.canvas.draw()
            if event.inaxes in [ax_transform]:
                if len(transform_circles) < 4:
                    circ = Circle((event.xdata, event.ydata), 10)
                    ax_transform.add_patch(circ)
                    transform_circles.append(circ)
                    fig.canvas.draw()


cid = fig.canvas.mpl_connect('button_press_event', onclick)

plt.show()
