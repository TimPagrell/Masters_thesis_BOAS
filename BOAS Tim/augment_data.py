import pandas as pd
from pathlib import Path
from time import time

from functions_and_classes import AudioUtil



# --------------
#   User input
# --------------

relative_folder = '' # '' if in same directory as this file
original_data_folder_name = 'BOAS-mobilljud'
augmented_data_folder_name = 'BOAS-mobilljud-augmented'
data_has_folds = False
n_augmentations = 2
do_eq = True

semitone_limits = [-1, 1]
stretch_rate_limits = [0.95, 1.05]

# Only equalization (make sure do_eq = True)
p_pitch_shift = 0
p_time_stretch = 0
do_at_least_one = False

# With these settings, we expect
# 1/3 with only pitch shift
# 1/3 with only time stretch
# 1/3 with both 
""" p_pitch_shift = 2/3
p_time_stretch = 0.5
do_at_least_one = True """

# With these settings, we expect
# 1/4 with only pitch shift
# 1/4 with only time stretch
# 1/4 with both
# 1/4 unchanged
""" p_pitch_shift = 0.5
p_time_stretch = 0.5
do_at_least_one = False """



# ------------------
#   Main execution
# ------------------

start = time()

# Create a folder for the augmented data if it does not already exist
augmented_folder_path = relative_folder + '/'*(len(relative_folder)>0) + augmented_data_folder_name
Path(augmented_folder_path).mkdir(parents=True, exist_ok=True)

# Read metadata file
original_folder_path = relative_folder + '/'*(len(relative_folder)>0) + original_data_folder_name
metadata_file = original_folder_path + '/metadata.csv'
df_metadata = pd.read_csv(metadata_file, encoding='utf-8-sig') # df = "DataFrame"

# Construct file path by concatenating fold and file name
if data_has_folds:
    df_metadata['file_name'] = df_metadata['slice_file_name']
    df_metadata['relative_path'] = '/fold' + df_metadata['fold'].astype(str) + '/' + df_metadata['file_name'].astype(str)
else:
    df_metadata['relative_path'] = '/' + df_metadata['file_name'].astype(str)

# Check if the metadata contains exercise test info
try: 
    df_metadata['exercise_test']
    et_exist = True
except: 
    et_exist = False

# Check if the file has info about creating folds
if 'folds' in df_metadata.columns:
    do_keep_folds = True

# Prepare lists for new metadata file
slice_file_name_list = []
augment_number_list = []
classID_list = []
pitch_shift_list = []
time_stretch_list = []
n_data = len(df_metadata)
if et_exist: et_list = []
if data_has_folds: fold_list = [] # Existing folds
if do_keep_folds: folds_list = [] # Folds to slice into in the future

# Augment all original data n_augmentations times 
for i_augment in range(n_augmentations+1):

    # Create new folder(s) for the augmented files, separating runs of the same original files
    current_augment_path = augmented_folder_path + f'/augment{i_augment}'
    Path(current_augment_path).mkdir(parents=True, exist_ok=True)

    for i_file in range(n_data):
        
        # Print progress
        progress = (i_augment-1)/n_augmentations + i_file/n_data/n_augmentations
        print(f'Progress: {progress*100:.2f}%  ', end='\r')

        # Find file of original audio data
        current_path = original_folder_path + '/' + df_metadata['relative_path'][i_file]
        
        # Prepare name and path for augmented data
        augmented_file_name = df_metadata['file_name'][i_file][:-4] + f'-aug{i_augment}' + '.wav'
        if data_has_folds:
            fold_i = df_metadata['fold'][i_file]
            augmented_file_path = current_augment_path + f'/fold{fold_i}'
            Path(augmented_file_path).mkdir(parents=True, exist_ok=True)
            augmented_file_path += '/' + augmented_file_name
        else:
            augmented_file_path = current_augment_path + '/' + augmented_file_name

        # Open audio file
        audio = AudioUtil.open(current_path)

        # Keep the first folder free of augmentation
        if i_augment == 0:
            AudioUtil.save(audio, augmented_file_path)
            shift_amount = 0
            stretch_rate = 0
        else:
            # Augment data
            augmented_audio, shift_amount, stretch_rate = \
            AudioUtil.offline_time_augment(audio, semitone_limits, p_pitch_shift, 
                                            stretch_rate_limits, p_time_stretch, 
                                            do_eq=do_eq, do_at_least_one=do_at_least_one, 
                                            do_print=False, return_settings=True)

            # Save augmented data in new folder
            AudioUtil.save(augmented_audio, augmented_file_path)

        # Append lists
        slice_file_name_list.append(augmented_file_name)
        augment_number_list.append(i_augment)
        classID_list.append(df_metadata['classID'][i_file])
        pitch_shift_list.append(shift_amount)
        time_stretch_list.append(stretch_rate)
        if et_exist: et_list.append(df_metadata['exercise_test'][i_file])
        if data_has_folds: fold_list.append(fold_i)
        if do_keep_folds: folds_list.append(df_metadata['folds'][i_file])

# Print 100% progress
print(f'Progress: 100%    ')
end = time()
print(f'Took {end-start:.2f}s')

# Turn list into dict and save as csv with pandas
augmented_dict = {'file_name': slice_file_name_list,
                'augment_number': augment_number_list,
                'classID': classID_list,
                'pitch_shift_amount': pitch_shift_list,
                'time_stretch_rate': time_stretch_list}
if et_exist: augmented_dict['exercise_test'] = et_list
if data_has_folds: augmented_dict['fold'] = fold_list
if do_keep_folds: augmented_dict['folds'] = folds_list
    

df_augmented = pd.DataFrame(augmented_dict)
df_augmented.to_csv(augmented_folder_path+'/metadata.csv', encoding='utf-8-sig', index=False)