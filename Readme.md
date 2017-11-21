[//]: # (Image References)

[image1]: ./output_images/undistorted_image.png "Undistorted"
[image2]: ./output_images/undistorted_image_2.png "Road Transformed"
[image3]: ./output_images/tuning_sobel.png "Tuning Sobel 1"
[image4]: ./output_images/tuning_sobel_2.png "Tuning Sobel 2"
[image5]: ./output_images/transformations.png "Transformations"
[image6]: ./output_images/combined.png "Combined"
[image7]: ./output_images/points_bird_eyes.png "Points of bird-eye view"
[image8]: ./output_images/bird_eyes_lines_view.png "Bird-eye view"


## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## [Rubric Points](https://review.udacity.com/#!/rubrics/571/view)

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it :smile:

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained file called `Correct.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I did some interactive images plots where I could tune the parameters of each binary transformation to get the best result.
Here is an example of tuning sobel in a test image.
![alt text][image3] ![alt text][image4]
You may take a look at this [video](https://youtu.be/DixUXMjzey4) to see how it works.
The code for tuning more parameters is in `InteractiveSobel.py` (for both orientation, x and y) and `InteractiveMag.py`. I did not use the direction of the gradient, because it did not improve finding lines.
The image below is an example of the output of the binary transformations.
![alt text][image5]
Moreover, I did a combination of this tree outputs, adding a [MORPH_CLOSE](https://docs.opencv.org/trunk/d9/d61/tutorial_py_morphological_ops.html) (go to section 4, this is provided by OpenCV) to get a better result of the lines.
![alt text][image6]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

To find the sources and destination points to create the bird-eye view perspective I did an interactive window of the algorithm, please see this [video](https://youtu.be/Ja9VKXalLBw) to have an idea how it works.
Te code is in file `InteractiveWarp.py`, here is the example how I did choose the points.
![alt text][image7]

The result points after many tests was:
| Source        | Destination   |
|:-------------:|:-------------:|
| 280, 670      | 270, 710      |
| 565, 470      | 270, 50       |
| 720, 470      | 1035, 50      |
| 1035, 670     | 1035, 710     |


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find the lane-line pixels I used the windowed plus histogram presented in the class. The code is in the file `MyHelpter.py` in the method `find_line` at line **87** and `find_line_with_last_data` at line **138**.
![alt text][image8]

In this method `find_line_with_last_data` at the line **165**, there is something really important to the code, the smallest radius that I assumed as possible to find the lines. I tunned it to 600 meters, in order to scape radius that did not work well because a failure in the identification of lines. If the radius is small then 600 meters I just skip it by using the averaged 10 last polynomials coefficients of the line.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To get the curvature, of both right and left lane-lines, and offset from the center I also used the example code provided in the class. You can see it at `MyHelpter.py` in the method `line_curvature_meters` at line **191** and in the method `center_offset_meters` at line **205**. The offset calculation was a bit more difficult to do, but I used the both lane-lines at some y point and the midpoint of the image 604 pixels, to get a variation in pixel, all that in pixel space. After that I converted to meters.

Generally, the curvature was about 1k for the curves and bigger than that for straight lines. Moreover the offset was about 10~40 cm.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

To convert back to the original image I used the class example as well. In addition I write down on the image frame the text for the curvatures, offset and which direction the car is turning :smile:
For that I used the second coefficient **b** of the line curvature polynomial `Ay² + by + c`.
Check the `MyHelpter.py` in the method `back_to_original` at the line 256!

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_out.mp4)
Also on the [YouTube](https://youtu.be/41aTByfd8So)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Well a had a lot of trouble to the class Lines (file `Lines.py`) working. Without that I would not complete the project, because at some points the lane-line detection just failed and I needed a backup for that frame, like the averaged last lines.
Moreover, I started with not that good bird-eye transformation, because of that I did the interactive window, to get in real time the changes that makes in the file and understand better it.
I am a hundred percent sure that my pipeline (file `Pipeline.py`) is going to fail on those challenges videos =(
I tried but with no luck, specially the hardest one where the curvature radius are small than 600 meters.
To get a robust pipeline I would have to work better in the identification of the lane-lines, such as using more algorithms or augmentation of the frames. Moreover, try another approach for harder curvatures radius.
