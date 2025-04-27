import os
import pandas as pd
from pathlib import Path
from time import time
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.stats import mannwhitneyu
import warnings

import sys

script_path = Path(__file__).resolve().parents[0]
BOAS_folder_path = Path(__file__).resolve().parents[1]
sys.path.append(str(BOAS_folder_path)) 

from functions_and_classes import AudioUtil, augment_for_feature_extraction

import opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)



# --------------
#   User input
# --------------

data_folder = 'BOAS-mobilljud'
use_sliced_data = True
use_augmented_data = True
model_suffix = ''
augmented_folder_suffix = ''
online_augments = 2

use_anova = False # If False, use Mann Whitney U test, otherwise use anova
filter_signal = True # If the adaptive high-pass filter should be used.
normalize_signal = False # 'RMS', 'Noise level' or False
p_threshold = 0.05
max_features = 100

use_2_classes = True # Simplifies the model from the four BOAS grades to only 2, BOAS negative (0&1) and BOAS positive (2&3)
use_exercise_test = False # Makes it so the model only trains on audio from after exercise test.
before_and_after_et = False # Overrides use_exercise_test. Makes it so the model trains on audio from both before and after exercise test, and any audio where that is uncertain.
make_et_class = False # only used if before_and_after_et = True, and overrides use_2_class. Makes it so the model now trains to determine if the audio is from before or after exercise test.



# ------------------
#   Main execution
# ------------------

model_suffix += augmented_folder_suffix

if use_augmented_data:
    data_folder += f'-augmented{augmented_folder_suffix}'

if use_sliced_data:
    data_folder += '-sliced'

# Import desired file
path = BOAS_folder_path/data_folder
df = pd.read_csv(path/'metadata.csv')

# Select audio files with 'exercise_test' = True or False depending on user input
if not before_and_after_et:
    df = df[df['exercise_test']==use_exercise_test].reset_index(drop=True)
elif make_et_class:
    df = df[[type(et) == type(True) for et in df['exercise_test']]]

# Change classID to 0&1=0 and 2&3=1
if make_et_class and before_and_after_et:
    df['classID'] = np.int64(df['exercise_test'])
elif use_2_classes:
    df['classID'] = np.int64(df['classID']>1)

# Find the name of each file
if use_augmented_data and use_sliced_data:
    name_of_file = 'augment' + df['augment_number'].astype(str) + '/' + 'fold' + df['fold'].astype(str) + '/'+ df['slice_file_name'].astype(str)
elif use_augmented_data:
    name_of_file = 'augment' + df['augment_number'].astype(str) + '/' + df['file_name'].astype(str)
elif use_sliced_data:
    name_of_file = 'fold' + df['fold'].astype(str) + '/' + df['slice_file_name'].astype(str)
else: 
    name_of_file = df['file_name'].astype(str)

features = []
start = time()
for i, file_name in enumerate(name_of_file):

    # Print progress
    progress = i/len(name_of_file)
    print(f'Feature extraction progress: {progress*100:.2f}%  ', end='\r')

    file_path = os.path.join(path, file_name)

    for i_aug in range(online_augments+1):

        # Process the file with OpenSmile

        if normalize_signal==False and not filter_signal and i_aug==0 :

            package_features = smile.process_file(file_path) # slower than process signal, but somehow yields better results

        else:
            
            signal, sr = AudioUtil.open(file_path)

            if filter_signal:
                signal, _ = AudioUtil.custom_high_pass_filter((signal, sr))

            if normalize_signal == 'RMS':
                signal, _ = AudioUtil.normalize_audio_by_RMS((signal, sr))
            elif normalize_signal == 'Noise level':
                signal, _ = AudioUtil.normalize_audio_by_noise_level((signal, sr))
                    
            if i_aug != 0:
                signal = augment_for_feature_extraction(signal, sr, output_numpy=False)

            signal = signal.numpy()
            
            package_features = smile.process_signal(signal, sr)

        # Convert dataframe row to dictionary and append it
        features.append(package_features.iloc[0].to_dict())

# Print 100% progress
print(f'Feature extraction progress: 100%    ')
print(f'Extracted {np.shape(package_features)[1]} features')
end = time()
print(f'Took {end-start:.2f}s')

# Create DataFrame from list of dictionaries
df_features = pd.DataFrame(features)

# Remove columns with constant values
df_features = df_features.loc[:, df_features.nunique() > 1]  

# Statistical test comparing features' class dependency
print('Performing statistical test')
X = df_features
y = df['classID'] # BOAS classID
y = pd.DataFrame({'classID': np.repeat(y, online_augments+1)}).reset_index(drop=True)['classID']
if use_anova:

    # ANOVA F-test (it will warn about constant features, so disable warnings just for this)
    warnings.simplefilter("ignore", UserWarning)
    anova = SelectKBest(f_classif, k='all')  # Use k='all' to keep all features
    anova.fit(X, y)
    warnings.simplefilter("default", UserWarning)

    # Get the ANOVA F-values and p-values for each feature
    anova_scores = anova.scores_  # F-values
    anova_pvalues = anova.pvalues_  # p-values

    # Create a DataFrame to display the feature scores and p-values
    anova_results = pd.DataFrame({
        'Feature': X.columns,
        'F-Value': anova_scores, # >10 - very good, <10 - could be good, <0 - bad
        'P-Value': anova_pvalues # <0.005 good
    })

    stat_results = anova_results

else:

    # Perform Mann-Whitney U-test for each feature
    p_values = []
    feature_names = []

    for feature in X.columns:
        group_0 = X[y == 0][feature]  # Values of feature for class 0
        group_1 = X[y == 1][feature]  # Values of feature for class 1

        if len(np.unique(group_0)) > 1 and len(np.unique(group_1)) > 1:  # Ensure variability
            stat, p = mannwhitneyu(group_0, group_1, alternative='two-sided')  # Mann-Whitney U-test
            p_values.append(p)
            feature_names.append(feature)

    # Create a DataFrame to display the feature scores and p-values
    U_results = pd.DataFrame({
        'Feature': feature_names,
        'P-Value': p_values 
    })

    stat_results = U_results

# Select the features with lowest P-value (below the given threshold, but no more than max_features)
chosen_features = stat_results[stat_results['P-Value']<p_threshold]
if len(chosen_features) > max_features:
    chosen_features = chosen_features.sort_values(by='P-Value', ascending=True).reset_index(drop=True)
    chosen_features = chosen_features.loc[range(max_features)]
chosen_features = chosen_features.reset_index(drop=True)

# Display the results
print(chosen_features)
print(f'Number of features: {len(chosen_features)}')

# Save the feature names to use when extracting features of individual files
slice_part_of_name = 'no'*(not use_sliced_data)+'slice'
aug_part_of_name = 'no'*(not use_augmented_data)+'aug'+(not not online_augments)*f'_{online_augments}'
et_part_of_name = f'{('no'*(not use_exercise_test))*(not before_and_after_et) + 'both'*before_and_after_et}et'
class_part_of_name = f'{2+2*(not use_2_classes)}class'*(not (make_et_class and before_and_after_et)) + 'etclass'*(make_et_class and before_and_after_et)
filter_part_of_name = 'no'*(not filter_signal)+'filter'
norm_part_of_name = 'RMSNorm'*(normalize_signal=='RMS') + 'NoiseNorm'*(normalize_signal=='Noise level') + 'noNorm'*(not normalize_signal)
variable_part_of_name = f'{slice_part_of_name}_{aug_part_of_name}_{et_part_of_name}_{class_part_of_name}_{filter_part_of_name}_{norm_part_of_name}'

save_path = script_path/'Feature data'/f'feature_names{model_suffix}_{variable_part_of_name}.csv'
chosen_features['Feature'].to_csv(save_path)

df_feature_names = pd.read_csv(save_path)['Feature']
print(df_features[df_feature_names])