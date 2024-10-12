# importing working and plotting libs
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import cv2
import data_extraction, image_processing, sorting_data


# finding desired data path and extracting at path
data_filename = '2006Aug2_083302'
data_path = os.path.join('C:\\Users\\Rares\\Desktop\\bubbles\\bubble_data\\Cordin 1 Prentice set\\Data Analysis Project - URG\\Best of Close to Sub\\0.2v parallel config\\',
                         data_filename)

# extracting at path
data = data_extraction.extract(data_path)

# processing image
keypoint_data = image_processing.process(data)

# sorting data
sorting_data.sort(keypoint_data)






