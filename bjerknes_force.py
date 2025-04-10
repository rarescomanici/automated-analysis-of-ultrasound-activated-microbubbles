import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# finds the bjerknes force for the first 8 frames of the sequence (t=0s -> t=6.31s) where oscillation is stable
def compute(data):

    # taking the first 8 records and saving them in a separate dataframe
    stable_frames = data[:8]

    # in order to get rest oscillation radii, we assume the lowest radius corresponds to the negative antinode
    # the highest to the positive antinode

    # we also assume ideal sinusoidal harmonic oscillations
    # so the initial radii/nodes would be equal to the mean of the distribution
    r_01 = stable_frames['Radius_1(microns)'].mean()
    r_02 = stable_frames['Radius_2(microns)'].mean()

    # to find the oscillation characteristics, we create 2 'variation' waves by subtracting the mean value
    # this brings the mean close to 0, and makes the movement easier to analyse
    wave_1 = stable_frames['Radius_1(microns)'] - r_01
    wave_2 = stable_frames['Radius_2(microns)'] - r_02

    # graphing wave
    # = sns.lineplot(x=stable_frames['Time(microsecs)'], y=wave_1,
                       #linestyle='--', marker='o')
    #ax1.set_title('Radial oscillations vs. time', fontdict={'size': 12, 'weight': 'bold'})
    #error1 = wave_1*0.2
    #ax1.errorbar(stable_frames['Time(microsecs)'], wave_1, yerr=np.abs(error1), fmt='o', color='b', alpha=0.5)
    #plt.show()

    # to find the amplitude, take the farthest antinode from the mean
    max_1, min_1 = wave_1.max(), wave_1.min()
    max_2, min_2 = wave_2.max(), wave_2.min()

    epsilon_1 = max(max_1, -min_1)
    epsilon_2 = max(max_2, -min_2)

    # normalising the waves
    wave_1 /= epsilon_1
    wave_2 /= epsilon_2

    # for the phase difference, the ongoing assumption is that both oscillations have similar frequencies
    # so we can find an absolute phase difference by averaging the phase differences between different points

    phi = np.arcsin(wave_2) - np.arcsin(wave_1)
    mean_phi = np.mean(phi)

    # check for overall phase difference, very close to 0
    #print(mean_phi)

    # medium parameters
    rho = 1000 # in kg/m^3
    omega = 1.267326e6 # camera frequency in Hz

    # insonation time until collapse can be found by taking the time between insonation starting (frame 10)
    # and when bubbles coalesce (for most bubble data, frame 23)
    dt = 10.25781843030128

    # finding dr_1, dr_2
    # assuming initial radii are the smallest
    dr_1, dr_2 = (max(data['Radius_1(microns)'])-r_01)/dt, (max(data['Radius_2(microns)'])-r_02)/dt

    # converting to SI
    r_01 /= 10e6
    r_02 /= 10e6
    d = data['Centroid_Distance(microns)']/10e6
    epsilon_1 /= 10e6
    epsilon_2 /= 10e6

    # we assume phase remains around the same value throughout the movement
    force_bj = -2*np.pi*rho*(omega**2)*(r_01**3)*(r_02**3)*epsilon_1*epsilon_2*np.cos(mean_phi)/(d**2)

    # and placing it onto the final dataset
    data.insert(10, 'Bjerknes_force(Newtons)', force_bj)

    return data