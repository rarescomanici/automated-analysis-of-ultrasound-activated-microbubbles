import glob
import cv2
import numpy as np

# Extracts data at path
def extract(data_path):

    data = []
    for image in glob.glob(data_path + "\\*.jpg"): # cropping images into 500x500 for efficiency, no features should be lost
        data.append(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)[250:750, 250:750]) # image is already grayscale, so it looks slightly weird when converted again
        # shouldn't interfere with blob detection
    return np.asarray(data) # returning np array for ease
