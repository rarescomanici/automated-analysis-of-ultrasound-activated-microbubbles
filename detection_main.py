# importing working and plotting libs
import os
import data_extraction, image_processing, sort_data, bjerknes_force
import pandas as pd
import numpy as np


# finding desired data path and extracting at path
data_path = 'C:\\Users\\Rares\\Desktop\\bubbles\\bubble_data\\Cordin 1 Prentice set\\Data Analysis Project - URG\\Best of Close to Sub\\0.2v perp config'
data_files = os.listdir(data_path)

os.makedirs('data', exist_ok=True) # creating output folder

fft_data = pd.DataFrame(columns=['Radius_1(microns)', 'Radius_2(microns)'])
for data_file in data_files: # looping through files

    path = os.path.join(data_path, data_file)

    # extracting at path
    data = data_extraction.extract(path)

    # processing image
    keypoint_data = image_processing.process(data)

    # sorting data
    final_data = sort_data.sort(keypoint_data)

    # bjerknes force analysis
    final_bjerknes = bjerknes_force.compute(final_data)

    # assembling radius fft data
    new_data = final_data[['Radius_1(microns)', 'Radius_2(microns)']].copy()
    fft_data = pd.concat([fft_data, new_data[:8]], axis=0, ignore_index=False)

    # writing dataframe to csv
    final_bjerknes.to_csv(f'data/{data_file}.csv', index=False)

camera_freq = 1.267326 # camera frequency in MHz
time_step = 1 / camera_freq # in microsecs

time = np.arange(fft_data.shape[0])*time_step
fft_data['Time(microsecs)'] = time

fft_data.to_csv(f'data/fft_data.csv', index=False)






