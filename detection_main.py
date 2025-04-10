# importing working and plotting libs
import os
import data_extraction, image_processing, sort_data, bjerknes_force
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pyplot as plt


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

# assembling time fft data
camera_freq = 1.267326 # camera frequency in MHz
time_step = 1 / camera_freq # in microsecs

fft_data = fft_data.dropna()
time = np.arange(fft_data.shape[0])*time_step
fft_data['Time(microsecs)'] = time

fft_data.to_csv(f'data/fft_data.csv', index=False)

#truncating fft data to n=6
fft_data_6 = fft_data[:64]
fft_r1 = scipy.fft.fft(fft_data_6['Radius_1(microns)'])

# single bubble data
#fft_data = pd.read_csv('C:\\Users\\Rares\\IdeaProjects\\automated-analysis-of-ultrasound-activated-microbubbles\\data\\set_2.csv')
#fft_data['Frequency(MHz)'] = 1/fft_data['Time(microsecs)']
#fft_data_6 = fft_data[:64]
#fft_r1 = scipy.fft.fft(fft_data_6['Radius_1(microns)'])

# converting to decibels according to 2.5 micron initial radius
fft_r1 = 10*np.log10((fft_r1/2.5)**2)

ax1 = sns.lineplot(fft_r1)
ax1.set_title('Mode power vs. frequency', fontdict={'size': 12, 'weight': 'bold'})
ax1.set_xlabel('Frequency(Hz)', fontdict={'size': 10})
ax1.set_ylabel('Power(dB)', fontdict={'size': 10})
plt.show()








