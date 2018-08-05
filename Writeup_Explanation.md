## Advanced Lane Finding Udacity SDND

### Gaurav Pokharkar

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

[//]: # (Image References)


![image2]: (/output_images/Original_vs_Undistorted_Warped.JPG) "Undistorted and Warped Image"
![image3]: (https://github.com/gpokhark/CarND-Advanced-Lane-Finding-Gaurav/tree/master/output_images/Perspective_View.JPG) "Thresholded and Perspective View"
![image4]: (https://github.com/gpokhark/CarND-Advanced-Lane-Finding-Gaurav/tree/master/output_images/Lane_Lines.JPG) "Lane_Lines"
![image5]: (https://github.com/gpokhark/CarND-Advanced-Lane-Finding-Gaurav/tree/master/output_images/Lane_Lines_Overlay.JPG) "Lane Lines Overlay"
![video1]: (https://github.com/gpokhark/CarND-Advanced-Lane-Finding-Gaurav/tree/master/output_images/project_video_output.mp4) "Video"

---

### Writeup / README


You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "Gaurav_Project.ipynb" 2nd block Camera Calibrate Function.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![](/output_images/Original_vs_UndistortedImage.jpg) "Original vs Undistorted Image"

Later stored the Camera Matrix and Distortion Coefficient in [wide_dist_pickle](https://github.com/gpokhark/CarND-Advanced-Lane-Finding-Gaurav/tree/master/output_images/camera_cal)

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The code for this step is contained in the first code cell of the IPython notebook located in "Gaurav_Project.ipynb" block Distortion Correction - Get Perspective Matrix.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]



#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is contained in the first code cell of the IPython notebook located in "Gaurav_Project.ipynb" block Individual HSL and Gradient Thresholding Function. This block has the basic functions.

In block Combined Thresholding Function I have applied saturation thresholding on the original image, light saturation on the original image, x direction sobel on s channel.

I used following combination to get vertical lines, eliminate horizontal lines and eliminate the shadow.

`# combined
combined = np.zeros_like(s_binary)
combined[((gradx == 1) | (s_binary == 1)) & ((l_binary == 1) & (s_binary == 1))] = 1`
    
Here's an example of my output for this step.
![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in the block - Perspective View Function `persp_view()`.
The `persp_view()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 320, 720      | 
| 1110, 720     | 920, 720      |
| 565, 470      | 320, 1        |
| 722, 470      | 920, 1        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used sliding window search to find an plot lane lines in my warped image. For this purpose I created a class file `lane_lines_1.py`. Created `find_lane_pixels()`, `measure_curvature_real` and `bad_lines`.

![alt text][image4]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature and position of the vehicle in the lane using `measure_curvature_real` function in the class file. I used the code taught in the video lectures.
`#Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension
#Calculation of R_curve (radius of curvature)
left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])`

Took the mean of the left and the right to get the curvature of the road. 

Later calculated the lane center by subtracting the fitted x and y values for the lane and subtracted if from the frame center point. This helped me calculate the position of the vehicle.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in block Complete Pipeline Structure lines # 45 through # 52 in my code in `Gaurav_Project.ipynb`.  
`#Unwarp the images
dstack_thresh = np.dstack((binary_w,binary_w,binary_w))*255
dstack_thresh_size = (dstack_thresh.shape[1], dstack_thresh.shape[0])
left_line_window = np.array(np.transpose(np.vstack([binary_warped.lx, binary_warped.ploty])))
right_line_window = np.array(np.flipud(np.transpose(np.vstack([binary_warped.rx, binary_warped.ploty]))))
line_points = np.vstack((left_line_window, right_line_window))
cv2.fillPoly(dstack_thresh, np.int_([line_points]), [0,255, 0])
un_warped = cv2.warpPerspective(dstack_thresh, persp_M_inv, dstack_thresh_size , flags=cv2.INTER_LINEAR)`

Here is an example of my result on a test image:

![alt text][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Pipeline for the video can be found in `Gaurav_pipeline.ipynb` file. It is combination of all the above explained functions.

Here's a [link to my video result][video1]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Problems faced -
1) Failure to detect lane lines in shadow and high light conditions
2) Failure to project lanes in absence of lane markings.

Solved the problems as follows - 
1) To eliminate shadow and light problems I used saturation thresholding on the original image, light saturation on the original image, x direction sobel on s channel as explained earlier.
2) Due to absence of right lane markings in certain frames I used the polynomial fit function from the previous frame.
3) In case of wobbly lines applied following condition to eliminate bad frames - 
`((new_mean > 1.3*old_mean) or (new_mean < 0.7*old_mean))`
mean - is the mean difference between the fitted x and fitted y values. Compared this value for the new frame and the old frame and when it was more than 1.3 times or less than 0.7 times the older width I neglected those frames.

To make my pipeline robust following improvements can be added -
1) Further more enhanced tuning of the thresholding function to clearly identify the lanes in high brightness and shadowy areas.
2) To add an algorithm that would help in sharp turning conditions.
3) An algorithm to take average of the last few frames and average it to avoid wobbly lines.
