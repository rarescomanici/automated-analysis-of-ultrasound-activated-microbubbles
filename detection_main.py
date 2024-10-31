# importing working and plotting libs
import os
import data_extraction, image_processing, sort_data


# finding desired data path and extracting at path
data_path = 'C:\\Users\\Rares\\Desktop\\bubbles\\bubble_data\\Cordin 1 Prentice set\\Data Analysis Project - URG\\Best of Close to Sub\\0.2v perp config'
data_files = os.listdir(data_path)

os.makedirs('data', exist_ok=True) # creating output folder

for data_file in data_files: # looping through files

    path = os.path.join(data_path, data_file)

    # extracting at path
    data = data_extraction.extract(path)

    # processing image
    keypoint_data = image_processing.process(data)

    # sorting data
    final_data = sort_data.sort(keypoint_data)

    # writing dataframe to csv
    final_data.to_csv(f'data/{data_file}.csv', index=False)






