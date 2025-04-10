import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# sorts relevant keypoint data into pandas dataframe
def sort(keypoints):

    # Creating dataframe and adding data
    keypoint_data = pd.DataFrame(columns=['Time(microsecs)', 'Radius_1(microns)', 'Radius_2(microns)', 'Volume_1(microns^3)', 'Volume_2(microns^3)',
                                          'X_1(microns)', 'X_2(microns)', 'Y_1(microns)', 'Y_2(microns)', 'Centroid_Distance(microns)'])

    for index in range(keypoints.shape[0]):

        keypoint = keypoints[index]
        if keypoint.shape[0] >= 2: # If there are more than 2 keypoints as desired, only the first 2 will be taken

            keypoint_data.loc[index, 'Radius_1(microns)'] = keypoint[0].size/2 # Keypoint size is the diameter, we divide by 2 to find radius
            keypoint_data.loc[index, 'Radius_2(microns)'] = keypoint[1].size/2
            keypoint_data.loc[index, 'X_1(microns)'] = keypoint[0].pt[0]
            keypoint_data.loc[index, 'X_2(microns)'] = keypoint[1].pt[0]
            keypoint_data.loc[index, 'Y_1(microns)'] = keypoint[0].pt[1]
            keypoint_data.loc[index, 'Y_2(microns)'] = keypoint[1].pt[1]

        elif keypoint.shape[0] == 1: # if there is only one keypoint, bind it to the closest previous neighbour

            # Taking the y coordinate as a better association feature

            if index == 0:

                keypoint_data.loc[index, 'Radius_1(microns)'] = keypoint[0].size/2
                keypoint_data.loc[index, 'Radius_2(microns)'] = 0
                keypoint_data.loc[index, 'X_1(microns)'] = keypoint[0].pt[0]
                keypoint_data.loc[index, 'X_2(microns)'] = 0
                keypoint_data.loc[index, 'Y_1(microns)'] = keypoint[0].pt[1]
                keypoint_data.loc[index, 'Y_2(microns)'] = 0

            elif np.abs(keypoint[0].pt[1] - keypoint_data.loc[index-1, 'Y_1(microns)']) < np.abs(keypoint[0].pt[1] - keypoint_data.loc[index-1, 'Y_2(microns)']):

                keypoint_data.loc[index, 'Radius_1(microns)'] = keypoint[0].size/2
                keypoint_data.loc[index, 'Radius_2(microns)'] = 0
                keypoint_data.loc[index, 'X_1(microns)'] = keypoint[0].pt[0]
                keypoint_data.loc[index, 'X_2(microns)'] = 0
                keypoint_data.loc[index, 'Y_1(microns)'] = keypoint[0].pt[1]
                keypoint_data.loc[index, 'Y_2(microns)'] = 0

            else:

                keypoint_data.loc[index, 'Radius_1(microns)'] = 0
                keypoint_data.loc[index, 'Radius_2(microns)'] = keypoint[0].size
                keypoint_data.loc[index, 'X_1(microns)'] = 0
                keypoint_data.loc[index, 'X_2(microns)'] = keypoint[0].pt[0]
                keypoint_data.loc[index, 'Y_1(microns)'] = 0
                keypoint_data.loc[index, 'Y_2(microns)'] = keypoint[0].pt[1]


        else: # If no points, set everything to 0

            keypoint_data.loc[index, 'Radius_1(microns)'] = 0
            keypoint_data.loc[index, 'Radius_2(microns)'] = 0
            keypoint_data.loc[index, 'X_1(microns)'] = 0
            keypoint_data.loc[index, 'X_2(microns)'] = 0
            keypoint_data.loc[index, 'Y_1(microns)'] = 0
            keypoint_data.loc[index, 'Y_2(microns)'] = 0

    scale = 8.25 # scaling factor px/micron
    camera_freq = 1.267326 # camera frequency in MHz
    keypoint_data = keypoint_data / scale
    time_step = 1 / camera_freq # in microsecs

    # Replacing 0 values with NaN for interpolation

    keypoint_data = keypoint_data.astype(float).replace(0, np.nan)

    # We use nearest neighbour interpolation to fill in missing data

    keypoint_data['Radius_1(microns)'] = keypoint_data['Radius_1(microns)'].interpolate(method='nearest', limit_direction='both')
    keypoint_data['Radius_2(microns)'] = keypoint_data['Radius_2(microns)'].interpolate(method='nearest', limit_direction='both')
    keypoint_data['X_1(microns)'] = keypoint_data['X_1(microns)'].interpolate(method='nearest', limit_direction='both')
    keypoint_data['Y_1(microns)'] = keypoint_data['Y_1(microns)'].interpolate(method='nearest', limit_direction='both')
    keypoint_data['X_2(microns)'] = keypoint_data['X_2(microns)'].interpolate(method='nearest', limit_direction='both')
    keypoint_data['Y_2(microns)'] = keypoint_data['Y_2(microns)'].interpolate(method='nearest', limit_direction='both')

    # Measurements

    d_sqr = ((keypoint_data['X_1(microns)']-keypoint_data['X_2(microns)'])**2 +
         (keypoint_data['Y_1(microns)']-keypoint_data['Y_2(microns)'])**2)
    keypoint_data['Centroid_Distance(microns)'] = np.sqrt(np.asarray(d_sqr, dtype="float64")) # Euclidean Distance

    keypoint_data['Volume_1(microns^3)'] = 4*np.pi*keypoint_data['Radius_1(microns)']**3/3 # Assumes bubbles are perfect spheres
    keypoint_data['Volume_2(microns^3)'] = 4*np.pi*keypoint_data['Radius_2(microns)']**3/3

    keypoint_data['Time(microsecs)'] = np.arange(keypoints.shape[0]) * time_step # Populating time column

    # standard deviation and mean
    sigma_1 = np.std(keypoint_data['Radius_1(microns)'])
    sigma_2 = np.std(keypoint_data['Radius_2(microns)'])
    sigma_3 = np.std(keypoint_data['Centroid_Distance(microns)'])

    mu_1 = np.mean(keypoint_data['Radius_1(microns)'])
    mu_2 = np.mean(keypoint_data['Radius_2(microns)'])
    mu_3 = np.mean(keypoint_data['Centroid_Distance(microns)'])

    # find outliers (more than 3 sigma away)
    outliers_1 = keypoint_data['Radius_1(microns)'][(keypoint_data['Radius_1(microns)'] < mu_1-3*sigma_1) | (keypoint_data['Radius_1(microns)'] > mu_1+3*sigma_1)].shape[0]
    outliers_2 = keypoint_data['Radius_2(microns)'][(keypoint_data['Radius_2(microns)'] < mu_2-3*sigma_2) | (keypoint_data['Radius_2(microns)'] > mu_2+3*sigma_2)].shape[0]
    outliers_3 = keypoint_data['Centroid_Distance(microns)'][(keypoint_data['Centroid_Distance(microns)'] < mu_3-3*sigma_3) | (keypoint_data['Centroid_Distance(microns)'] > mu_3+3*sigma_3)].shape[0]

    # find CI by checking 2 sigma; the signal lasts until frame 64
    CI_1 = (64-keypoint_data['Radius_1(microns)'][(keypoint_data['Radius_1(microns)'] < mu_1-2*sigma_1) | (keypoint_data['Radius_1(microns)'] > mu_1+2*sigma_1)].shape[0])/64
    CI_2 = (64-keypoint_data['Radius_2(microns)'][(keypoint_data['Radius_2(microns)'] < mu_2-2*sigma_2) | (keypoint_data['Radius_2(microns)'] > mu_2+2*sigma_2)].shape[0])/64
    CI_3 = (64-keypoint_data['Centroid_Distance(microns)'][(keypoint_data['Centroid_Distance(microns)'] < mu_3-2*sigma_3) | (keypoint_data['Centroid_Distance(microns)'] > mu_3+2*sigma_3)].shape[0])/64

    #print(sigma_3, CI_3)

    # plot
    #ax1 = sns.lineplot(data=keypoint_data, x=keypoint_data['Time(microsecs)'], y=keypoint_data['Radius_1(microns)'],
                      #linestyle='--', marker='o')
    #error1 = np.ones(keypoint_data['Radius_1(microns)'].shape)/scale+keypoint_data['Radius_1(microns)']*0.05
    #ax1.errorbar(keypoint_data['Time(microsecs)'], keypoint_data['Radius_1(microns)'], yerr=error1, fmt='o', color='b', alpha=0.5)
    #ax1.set_title('Radius vs. time', fontdict={'size': 12, 'weight': 'bold'})

    #ax2 = sns.lineplot(data=keypoint_data, x=keypoint_data['Time(microsecs)'], y=keypoint_data['Radius_2(microns)'],
                       #linestyle='--', marker='o')
    #error2 = np.ones(keypoint_data['Radius_2(microns)'].shape)/scale+keypoint_data['Radius_2(microns)']*0.05
    #ax2.errorbar(keypoint_data['Time(microsecs)'], keypoint_data['Radius_2(microns)'], yerr=error2, fmt='o', color='r', alpha=0.5)
    #ax2.set_title('Radius vs. time', fontdict={'size': 12, 'weight': 'bold'})

    #ax3 = sns.lineplot(data=keypoint_data, x=keypoint_data['Time(microsecs)'], y=keypoint_data['Centroid_Distance(microns)'],
                       #linestyle='--', marker='o')
    #error3 = np.ones(keypoint_data['Centroid_Distance(microns)'].shape)/scale+keypoint_data['Centroid_Distance(microns)']*0.05
    #ax3.errorbar(keypoint_data['Time(microsecs)'], keypoint_data['Centroid_Distance(microns)'], yerr=error3, fmt='o', color='b', alpha=0.5)
    #ax3.set_title('Centroid distance vs. time', fontdict={'size': 12, 'weight': 'bold'})

    plt.show()

    return keypoint_data
