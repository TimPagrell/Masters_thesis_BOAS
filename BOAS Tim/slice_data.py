import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from time import time

from functions_and_classes import AudioUtil, calculate_fold_timings, calculate_slice_timings



# --------------
#   User input
# --------------

relative_folder = '' # '' if in same directory as this file
original_data_parent_folder_name = 'BOAS-mobilljud'
sliced_data_parent_folder_name = original_data_parent_folder_name+ '-sliced'
slice_jump = 1 # in seconds
slice_length = 6 # in seconds
data_is_augmented = False



# ------------------
#   Main execution
# ------------------

# Create a folder for the sliced data if it does not already exist
sliced_parent_folder_path = relative_folder + '/'*(len(relative_folder)>0) + sliced_data_parent_folder_name
Path(sliced_parent_folder_path).mkdir(parents=True, exist_ok=True)

# Read metadata file
original_parent_folder_path = relative_folder + '/'*(len(relative_folder)>0) + original_data_parent_folder_name
metadata_file = original_parent_folder_path + '/metadata.csv'
df_metadata = pd.read_csv(metadata_file, encoding='utf-8-sig') # df = "DataFrame"

# Print filenames
""" for i in range(len(df_metadata.index)):
    print(df_metadata['file_name'][i])
print() """

# Check if the metadata contains exercise test info
try: 
    df_metadata['exercise_test']
    et_exist = True
except: 
    et_exist = False

# Determine how many augment folders there are. 1 if none
if data_is_augmented:
    n_augments = len(list(set(df_metadata['augment_number'])))
else:
    n_augments = 1

start = time()

# Prepare lists for the new metadata file
slice_file_name_list = []
fold_list = []
classID_list = []
start_time_list = []
end_time_list = []
if et_exist: et_list = []
if data_is_augmented: augment_number_list = []

# Run for all augment folders (only once if none)
for i_augment in range(n_augments):

    if data_is_augmented:
        
        # Select only the metadata of the correct augment folder
        augment_df_mask = df_metadata['augment_number'].isin([i_augment])
        df_current_metadata = df_metadata[augment_df_mask].reset_index(drop=True)
        
        # Create new folder(s) for the augmented files, separating runs of the same original files
        original_folder_path = original_parent_folder_path + f'/augment{i_augment}'
        sliced_data_folder_name = sliced_data_parent_folder_name + f'/augment{i_augment}'
        sliced_folder_path = sliced_parent_folder_path + f'/augment{i_augment}'
        Path(sliced_data_folder_name).mkdir(parents=True, exist_ok=True)

    else:
        original_folder_path = original_parent_folder_path
        sliced_data_folder_name = sliced_data_parent_folder_name
        sliced_folder_path = sliced_parent_folder_path
        df_current_metadata = df_metadata

    # Loop over every entry in the metadata file
    n_data = len(df_current_metadata)
    for i_file in range(n_data):

        # Print progress
        progress = ((i_file)/n_data + (i_augment-1))/n_augments
        print(f'Progress: {progress*100:.2f}%  ', end='\r')

        # Reset slice_id to 1, so that each files slices starts counting from 1
        slice_id = 1
        # Convert the 'folds' column entry into a list
        folds = np.int32(list(df_current_metadata['folds'][i_file].replace(' ', '').split(',')))

        # Open audio file
        current_path = original_folder_path+'/'+df_current_metadata['file_name'][i_file]
        audio = AudioUtil.open(current_path)
        signal, sr = audio

        # Caculate fold start and end times
        fold_start_times, fold_end_times = calculate_fold_timings(signal, folds, slice_jump, sr)

        # Loop over each fold
        for i_fold, fold in enumerate(folds):

            # Create new fold folder if needed
            current_slice_path = sliced_folder_path + f'/fold{fold}'
            Path(current_slice_path).mkdir(parents=True, exist_ok=True)

            # Calculate slice start and end times
            slice_start_times, slice_end_times = \
            calculate_slice_timings(fold_start_times[i_fold], fold_end_times[i_fold], slice_jump, slice_length)

            # Loop over every slice in the fold
            for i_slice in range(len(slice_start_times)):

                # Prepare name and path for sliced data
                sliced_file_name = df_current_metadata['file_name'][i_file][:-4] + f'-slice{slice_id}' + '.wav'
                sliced_file_path = current_slice_path + '/' + sliced_file_name

                # Slice the signal
                start_time = slice_start_times[i_slice]
                end_time = slice_end_times[i_slice]
                sliced_signal = signal[:,int(start_time*sr):int(end_time*sr)]
                sliced_audio = (sliced_signal, sr)

                # Save sliced data
                AudioUtil.save(sliced_audio, sliced_file_path)

                # Append lists
                slice_file_name_list.append(sliced_file_name)
                fold_list.append(fold)
                classID_list.append(df_current_metadata['classID'][i_file])
                start_time_list.append(start_time)
                end_time_list.append(end_time)
                if et_exist: et_list.append(df_current_metadata['exercise_test'][i_file])
                if data_is_augmented: augment_number_list.append(i_augment)

                # Increment slice_id
                slice_id += 1

# Print 100% progress
print(f'Progress: 100%    ')
end = time()
print(f'Took {end-start:.2f}s')

# Turn list into dict and save as csv with pandas
sliced_dict = {'slice_file_name': slice_file_name_list,
            'fold': fold_list,
            'classID': classID_list,
            'start_time': start_time_list,
            'end_time': end_time_list}
if et_exist: sliced_dict['exercise_test'] = et_list
if data_is_augmented: sliced_dict['augment_number'] = augment_number_list

df_augmented = pd.DataFrame(sliced_dict)

df_augmented.to_csv(sliced_parent_folder_path+'/metadata.csv', encoding='utf-8-sig', index=False)