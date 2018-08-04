import numpy as np
import os
import cv2
import glob
import matplotlib.pyplot as plt
import pickle
import matplotlib.image as mpimg

class Lane_Lines(object):
    
    # Constructor
    def __init__(self, binary_warped,plf,prf):
        self.binary_warped = binary_warped
        self.nwindows = 10
#         self.left_fitx = []
#         self.right_fitx = []
        self.ly = []
        self.lx = []
        self.ry = []
        self.rx = []
        self.ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0])
        self.left_fit = plf
        self.right_fit = prf
        
    def find_lane_pixels(self):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(self.binary_warped[self.binary_warped.shape[0]//2:,:], axis=0)
        # Create an output image to draw on and visualize the result
        out_img = np.dstack((self.binary_warped, self.binary_warped, self.binary_warped))
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
#         print(leftx_base)
#         print(rightx_base)

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
        non_zero = self.binary_warped.nonzero()
        non_zeroy = np.array(non_zero[0])
        non_zerox = np.array(non_zero[1])
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
            good_left_inds = ((non_zeroy >= win_y_low) & (non_zeroy < win_y_high) & (non_zerox >= win_xleft_low) & \
                              (non_zerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((non_zeroy >= win_y_low) & (non_zeroy < win_y_high) & (non_zerox >= win_xright_low) & \
                           (non_zerox < win_xright_high)).nonzero()[0]
        
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
#             print('Left Lane Inds', len(left_lane_inds))
#             print('Right Lane Inds', len(right_lane_inds))
        
            ### TO-DO: If you found > minpix pixels, recenter next window ###
            ### (`right` or `leftx_current`) on their mean position ###
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(non_zerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(non_zerox[good_right_inds]))
            


        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = non_zerox[left_lane_inds]
        lefty = non_zeroy[left_lane_inds] 
        rightx = non_zerox[right_lane_inds]
        righty = non_zeroy[right_lane_inds]
        
        badlines = False
        
        if (leftx.size == 0 and lefty.size == 0):
            badlines = True
        if (rightx.size == 0 and righty.size == 0):
            badlines = True
        if (np.mean(leftx) > np.mean(rightx)):
            badlines = True
        t_left_fit = self.left_fit
        t_right_fit = self.right_fit
        
        if not badlines:
#             self.left_fit = np.polyfit(lefty, leftx, 2)
#             self.right_fit = np.polyfit(righty, rightx, 2)
            t_left_fit = np.polyfit(lefty, leftx, 2)
            t_right_fit = np.polyfit(righty, rightx, 2)            

        # Generate x and y values for plotting
#         ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0] )
#         try:
#             left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
#             right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
        left_fitx = t_left_fit[0]*self.ploty**2 + t_left_fit[1]*self.ploty + t_left_fit[2]
        right_fitx = t_right_fit[0]*self.ploty**2 + t_right_fit[1]*self.ploty + t_right_fit[2]
#         except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
#             print('The function failed to fit a line!')
#             left_fitx = 1*self.ploty**2 + 1*self.ploty
#             right_fitx = 1*self.ploty**2 + 1*self.ploty
            

        badlines = self.bad_lines(left_fitx,right_fitx,leftx,lefty,rightx,righty,t_left_fit,t_right_fit)
        
        
        if not badlines:
            self.left_fit = np.polyfit(lefty, leftx, 2)
            self.right_fit = np.polyfit(righty, rightx, 2)
            
            
#         try:
        left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2]
#         except TypeError:
#             # Avoids an error if `left` and `right_fit` are still none or incorrect
#             print('The function failed to fit a line!')
#             left_fitx = 1*self.ploty**2 + 1*self.ploty
#             right_fitx = 1*self.ploty**2 + 1*self.ploty       

        ## Visualization ##
        # Colors in the left and right lane regions
        out_img[lefty, leftx] = [255, 0, 0]
        out_img[righty, rightx] = [0, 0, 255]

        # Plots the left and right polynomials on the lane lines
#         plt.plot(left_fitx, self.ploty, color='yellow')
#         plt.plot(right_fitx, self.ploty, color='yellow')

        
#         self.ly = np.copy(lefty)
        self.lx = np.copy(left_fitx)
#         self.ry = np.copy(righty)
        self.rx = np.copy(right_fitx)
        return out_img, self.left_fit, self.right_fit
    
    def measure_curvature_real(self):
        '''
        Calculates the curvature of polynomial functions in meters.
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
        # Start by generating our fake example data
        # Make sure to feed in your real data instead in your project!
#         ploty = np.linspace(0, self.binary_warped.shape[0]-1, self.binary_warped.shape[0] )
        left_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.lx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(self.ploty*ym_per_pix, self.rx*xm_per_pix, 2)
    
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(self.ploty)
    
        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        # Lane Center
        lane_center = (self.rx[self.binary_warped.shape[0]-1]-self.lx[self.binary_warped.shape[0]-1])/2
        center_offset_p = abs(lane_center - ((self.binary_warped.shape[0])/2))
        center_offset_m = center_offset_p * xm_per_pix
    
        return left_curverad, right_curverad, center_offset_m
        
    
    def bad_lines(self,left_fitx,right_fitx,leftx,lefty,rightx,righty,t_left_fit,t_right_fit):
        if (self.left_fit == [] and self.right_fit == []):
            return False
        else:
            old_left_fitx = self.left_fit[0]*self.ploty**2 + self.left_fit[1]*self.ploty + self.left_fit[2]
            old_right_fitx = self.right_fit[0]*self.ploty**2 + self.right_fit[1]*self.ploty + self.right_fit[2] 
            old_mean = np.mean(old_right_fitx-old_left_fitx)
            
            new_mean = np.mean(right_fitx-left_fitx)
            
            slope = 2*self.right_fit[0]*self.ploty + self.right_fit[1]
            
            if ((t_left_fit == []) and (t_right_fit == [])):
                return True
            if (leftx.size == 0 and lefty.size == 0):
                return True
            if (rightx.size == 0 and righty.size == 0):
                return True
            if (np.mean(leftx) > np.mean(rightx)):
                return True
            if ((new_mean > 1.3*old_mean) or (new_mean < 0.7*old_mean)):
                return True
            
            if (np.mean(right_fitx) < np.mean(left_fitx)):
                return True
            
            if (right_fitx[-1] <= self.binary_warped.shape[1]/2):
                return True
            
            if (right_fitx[-1] >= 1000):
                return True

