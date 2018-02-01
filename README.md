
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

[//]: # (Image References)

[image01]: ./output_images/Sobel_Threshold.png 
[image02]: ./output_images/Sobel_Threshold_1.png 
[image03]: ./output_images/Sobel_Threshold_2.png 
[image04]: ./output_images/Sobel_Threshold_3.png 

[image1]: ./output_images/Transform.png


[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./output_images/test_output.png 
[video1]: ./output_images/out.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 
You're reading it!

### Camera Calibration

#### 1. Camera matrix, distortion coefficients and distortion correction.

The code for this step is contained in the first code cell of the IPython notebook located in `./example.ipynb`. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 
`
Camera matrix and distortion coefficient stored successful:
[[  1.15396093e+03   0.00000000e+00   6.69705359e+02]
 [  0.00000000e+00   1.14802495e+03   3.85656232e+02]
 [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
[[ -2.41017968e-01  -5.30720497e-02  -1.15810318e-03  -1.28318543e-04
    2.67124302e-02]]
`
<img src="./camera_cal/calibration1.jpg" width="500">
<img src="./camera_cal/1_undistort_calibration1.jpg" width="500">
<h4 align = "center">Chessboard used for camera calibration.</h4>

saved as `calibrate_camera_matrix.p` for further usage.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

<img src="./test_images/test5.jpg" width="500">
<h4 align = "center">Before</h4>
<img src="./output_images/21_undistort_test5.jpg" width="500">
<h4 align = "center">Distortion Corrected</h4>


#### 2. Sobel Threshold.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at cell 3 in `./example.ipynb`). 

* abs_sobel_thresh `cv2.Sobel(gray, cv2.CV_64F, 1, 0)`
* mag_thresh 
     
    `sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
     gradmag = np.sqrt(sobelx**2 + sobely**2)`    
* dir_threshold
     
     `absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))`

* hls_thresh
     `
     hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
     `
* combined_thresh combine the above threshold

Here's an example of my output for this step.

![alt text][image01]
![alt text][image02]
![alt text][image03]
![alt text][image04]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform`, which appears in cell 4 in `./example.ipynb`.  The function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image1]

#### 4. Describe how you identified lane-line pixels and fit their positions with a polynomial?

I did this step in cell 5 in my code in `./example.ipynb`. Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this step in cell 6 in my code in `./example.ipynb`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in cell 6 in my code in `./example.ipynb`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


