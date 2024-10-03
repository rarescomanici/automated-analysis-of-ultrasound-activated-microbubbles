# importing working and plotting libs
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import glob
from PIL import Image
import scipy
import skimage
import cv2


# finding desired data path and all files of chosen extension
data_filename = '2006Aug2_083302'
data_path = os.path.join('C:\\Users\\Rares\\Desktop\\bubbles\\bubble_data\\Cordin 1 Prentice set\\Data Analysis Project - URG\\Best of Close to Sub\\0.2v parallel config\\',
                         data_filename)
data = []
for image in glob.glob(data_path + "\\*.jpg"): # cropping images into 500x500 for efficiency, no features should be lost
    data.append(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)[250:750, 250:750]) # image is already grayscale, so it looks slightly weird when converted again
    # shouldn't interfere with edge detection
data = np.asarray(data)

# processing image

processing_data = data.copy() # saving for data integrity

threshold = 200 # threshold for binarisation, possible param for ML training
maxval = 255 # max pixel value for grayscale

## Setup SimpleBlobDetector parameters
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = maxval-1

# Filter by Color
params.filterByColor = True
params.blobColor = 255 # detecting white in binary mask

# Filter by Area
params.filterByArea = True
params.minArea = 300 # this works with most blobs
params.maxArea = 50000 # as big as possible without going into background

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.5

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Set up the detector with parameters.
detector = cv2.SimpleBlobDetector_create(params)

for image_index in range(processing_data.shape[0]):

    mean = np.mean(processing_data[image_index]) # finding the mean to adapt to changes in brightness, contrast for now - possible ML training param
    [alpha, beta, gamma] = [1/mean, 0.0, 0.4] # adjustments for contrast, brightness, gamma correction
    mask = cv2.medianBlur(processing_data[image_index], 3) # median filter function does not work on 3d arrays
    # the array needs to be iterated through
    mask = mask**gamma * alpha + beta # masking values with alpha, beta, gamma
    # beta param does not do anything when the data is normalized so it is set to 0 for now

    ## Blob detection approach
    im_in = ((mask / mask.max())*255).astype(np.uint8) # normalizing
    th, im_th = cv2.threshold(im_in, threshold, maxval, cv2.THRESH_BINARY_INV)

    # Copy the thresholded image.
    im_floodfill = im_th.copy()

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    im_mask = np.zeros((h+2, w+2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, im_mask, (0,0), 255);

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = im_th | im_floodfill_inv

    # Detect blobs.
    keypoints = detector.detect(im_out) # keypoint stores (x,y), size, angle etc.

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(im_out, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)

    ## HoughCircle approach, not useful yet
    # final = cv2.normalize(src=mask, dst=None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) # normalization
    # detected_circles = cv2.HoughCircles(final,
    #                                     cv2.HOUGH_GRADIENT, 1, mask.shape[0]/16, param1 = 100,
    #                                     param2 = 30, minRadius = 0, maxRadius = 0)
    # if detected_circles is not None:
    #     # Convert the circle parameters a, b and r to integers.
    #     detected_circles = np.uint16(np.around(detected_circles))
    #
    #     for pt in detected_circles[0, :]:
    #         a, b, r = pt[0], pt[1], pt[2]
    #
    #         # Draw the circumference of the circle.
    #         cv2.circle(processing_data[image_index], (a, b), r, (0, 255, 0), 2)
    #
    #         # Draw a small circle (of radius 1) to show the center.
    #         cv2.circle(processing_data[image_index], (a, b), 1, (0, 0, 255), 3)
    #         cv2.imshow("Detected Circle", processing_data[image_index])
    #         cv2.waitKey(0)

    #img_plot = plt.imshow(im_out)
    #plt.show()

#print(processing_data[0])




