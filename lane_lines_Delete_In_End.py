import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg

class Lane_Lines(object):
    
    # Constructor
    def __init__(self, binary_warped):
        self.binary_warped = binary_warped
        self.nwindows = 10
        self.left_fitx = []
        self.right_fitx = []
        self.ly = []
        self.lx = []
        self.ry = []
        self.rx = []
        
    def find_lane_pixels(self):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))
    #     out_img = np.copy(binary_warped)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # HYPERPARAMETERS
        # Choose the number of sliding windows
        self.nwindows = 10
        # Set the width of the windows +/- margin
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 100

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(self.binary_warped.shape[0]//self.nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(self.nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.binary_warped.shape[0] - (window+1)*window_height
            win_y_high = self.binary_warped.shape[0] - window*window_height
            ### TO-DO: Find the four below boundaries of the window ###
            win_xleft_low = leftx_current - margin  # Update this
            win_xleft_high = leftx_current + margin  # Update this
            win_xright_low = rightx_current - margin  # Update this
            win_xright_high = rightx_current + margin  # Update this
        
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),\
                          (win_xleft_high,win_y_high),(0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),\
                          (win_xright_high,win_y_high),(0,255,0), 2) 
        
            ### TO-DO: Identify the nonzero pixels in x and y within the window ###
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & \
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & \
                           (nonzerox < win_xright_high)).nonzero()[0]
        
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
#             print(len(left_lane_inds))
#             print(len(right_lane_inds))
        
#             print(len(good_left_inds))
#             print(len(good_right_inds))
        
            ### TO-DO: If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
#             pass # Remove this when you add your function

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
#         print('This')
#         print(len(leftx),len(rightx))
#         print(len(lefty),len(righty))
#         print('This')
        return leftx, lefty, rightx, righty, out_img


    def fit_polynomial(self):
        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]//2:,:], axis=0)
        # Find our lane pixels first
        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels()
    #     print(len(leftx),len(rightx))
        if len(rightx) == 0:
            midpoint = np.int(histogram.shape[0]//2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            rightx = leftx + 1*(rightx_base - leftx_base)
        elif len(leftx) == 0:
            midpoint = np.int(histogram.shape[0]//2)
            leftx_base = np.argmax(histogram[:midpoint])
            rightx_base = np.argmax(histogram[midpoint:]) + midpoint
            leftx = rightx - 1*(rightx_base - leftx_base)
        if len(righty) == 0:
            righty = lefty
        elif len(lefty) == 0:
            lefty = righty
            
        ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
        if (len(rightx) > 0 and len(leftx) > 0):
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0] )
        try:
            self.left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            self.right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            self.left_fitx = 1*ploty**2 + 1*ploty
            self.right_fitx = 1*ploty**2 + 1*ploty

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
        plt.plot(self.left_fitx, ploty, color='yellow')
        plt.plot(self.right_fitx, ploty, color='yellow')

        
        self.ly = np.copy(lefty)
        self.lx = np.copy(leftx)
        self.ry = np.copy(righty)
        self.rx = np.copy(rightx)
        return out_img
    
    def measure_curvature_real(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
        # Start by generating our fake example data
        # Make sure to feed in your real data instead in your project!
        ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0] )
        left_fit_cr = np.polyfit(self.ly*ym_per_pix, self.lx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ry*ym_per_pix, self.rx*xm_per_pix, 2)
    
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)
    
        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
        return left_curverad, right_curverad