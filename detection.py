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

threshold = 35 # threshold for binarisation, possible param for ML training
maxval = 255 # max pixel value for grayscale

for image_index in range(processing_data.shape[0]):
    binary_mask = (cv2.medianBlur(processing_data[image_index], 3) < threshold) * maxval # median filter function does not work on 3d arrays
    # the array needs to be iterated through
    #detected_circles = cv2.HoughCircles(binary_mask,
                                        #cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,
                                        #param2 = 30, minRadius = 1, maxRadius = 40)

    img_plot = plt.imshow(binary_mask)
    plt.show()

#print(processing_data[0])




