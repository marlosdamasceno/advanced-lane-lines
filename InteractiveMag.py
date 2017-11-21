import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.image as mpimg
from Helper import mag_thresh

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(left=0.25, bottom=0.25)
min0 = 0
max0 = 255

image = mpimg.imread('TestImages/test1.jpg')
im = mag_thresh(image, sobel_kernel=3, threshold=(min0, max0), channel=2)
im1 = ax.imshow(im, cmap='gray')
fig.colorbar(im1)

axcolor = 'lightgoldenrodyellow'
axmin = fig.add_axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axmax = fig.add_axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

smin = Slider(axmin, 'Min', 0, max0, valinit=min0, valfmt='%0.0f')
smax = Slider(axmax, 'Max', 0, max0, valinit=max0, valfmt='%0.0f')


def update(val):
    im1.set_data(mag_thresh(image, sobel_kernel=3, threshold=(smin.val, smax.val), channel=2))
    fig.canvas.draw()


smin.on_changed(update)
smax.on_changed(update)
plt.show()
