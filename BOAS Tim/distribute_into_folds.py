import pandas as pd
from pathlib import Path
import numpy as np
np.set_printoptions(precision=3)

import sys

BOAS_folder_path = Path(__file__).resolve().parents[0]


# --------------
#   User input
# --------------

data_folder = 'BOAS-mobilljud'
K = 5 # K in "K fold cross-validation"



# ------------------
#   Main execution
# ------------------

# Read metadata file
path = BOAS_folder_path/data_folder
df = pd.read_csv(path/'metadata.csv')

# Remove any existing folds column
try:
    df = df.drop(columns=['folds'])
except:
    pass

# Prepare lists
n_entries = len(df)
folds_list = []
folds_counter = np.zeros([K,8]) # Two counters per BOAS grade (before and after ET), per fold

# Initial distribution of folds: one fold per file
for i_file in range(n_entries):

    classID = df['classID'][i_file] # BOAS grade 
    exercise_test = df['exercise_test'][i_file] # ET bool
    
    # Find the folds that are next in turn to recieve a file of the current BOAS grade and ET bool
    all_folds_counter = folds_counter[:,classID+4*exercise_test]
    candidate_folds = np.arange(K)[(all_folds_counter==np.min(all_folds_counter))]

    # Out of these folds, find which folds in the corresponding BOAS grade of the same BOAS positive vs negative class 
    # that has fewest files (ex: if classID=1, check fold counters of classID=0, and if classID=2, check classID=3 etc.)
    other_classID = classID + (classID+1)%2 - classID%2 # 0->1, 1->0, 2->3, 3->2
    folds_of_other_classID_counter = folds_counter[candidate_folds, other_classID+4*exercise_test]
    candidate_folds = candidate_folds[(folds_of_other_classID_counter==np.min(folds_of_other_classID_counter))]

    # Select randomly out of those folds which fold to add the file to
    fold = np.random.choice(candidate_folds)

    # Add the file and update counters
    folds_list.append(str(fold+1))
    folds_counter[fold, classID+4*exercise_test] += 1

df['folds'] = folds_list

folds_counter_original = folds_counter.copy()

# Further distribution of 'unfinished' folds, further evening out the differences between the number of files per fold 
# USED ONLY WHEN SLICING DATA
for i, counter in enumerate(np.sum(folds_counter, axis=0)):
    
    classID = i%4
    exercise_test = i>=4
    counter = counter%K

    while counter != 0:
        
        # Find the folds that are next in turn to recieve and share a file of the current BOAS grade and ET bool
        all_folds_counter = folds_counter[:,classID+4*exercise_test]
        candidate_recieving_folds = np.arange(K)[(all_folds_counter==np.min(all_folds_counter))]
        candidate_sharing_folds = np.arange(K)[(all_folds_counter==np.max(all_folds_counter))]

        # Out of these folds, find which folds in the corresponding BOAS grade of the same BOAS positive vs negative class 
        # that has fewest/most files (ex: if classID=1, check fold counters of classID=0, and if classID=2, check classID=3 etc.)
        other_classID = classID + (classID+1)%2 - classID%2 # 0->1, 1->0, 2->3, 3->2
        receiving_folds_of_other_classID_counter = folds_counter[candidate_recieving_folds, other_classID+4*exercise_test]
        sharing_folds_of_other_classID_counter = folds_counter[candidate_sharing_folds, other_classID+4*exercise_test]
        candidate_recieving_folds = candidate_recieving_folds[(receiving_folds_of_other_classID_counter==np.min(receiving_folds_of_other_classID_counter))]
        candidate_sharing_folds = candidate_sharing_folds[(sharing_folds_of_other_classID_counter==np.max(sharing_folds_of_other_classID_counter))]

        # Select randomly out of those folds which fold to add the file to, and which fold to share from, respectively
        recieving_fold = np.random.choice(candidate_recieving_folds)
        sharing_fold = np.random.choice(candidate_sharing_folds)
        
        # Select random file in the chosen sharing fold, and append the recieving fold to the 'folds' element
        df_current = df[df['exercise_test']==exercise_test]
        df_current = df_current[df_current['classID']==classID]
        df_current = df_current[[folds[0]==str(sharing_fold+1) for folds in df_current['folds'].values]]
        len_of_folds = [len(folds) for folds in df_current['folds'].values]
        
        df_current = df_current[len_of_folds==np.min(len_of_folds)]

        sampled_index = df_current.sample().index[0]

        df.loc[sampled_index, 'folds'] += ', ' + str(recieving_fold+1)

        current_folds = np.int32(df.loc[sampled_index, 'folds'].replace(' ','').split(','))
        num_folds = len(current_folds)
        for i_fold, fold in enumerate(current_folds):
            fold = fold-1
            if i_fold == 0:
                folds_counter[fold, classID+4*exercise_test] = folds_counter_original[fold, classID+4*exercise_test] - 0.9999*((len(df.loc[sampled_index, 'folds'])-1)//3)/(1+(len(df.loc[sampled_index, 'folds'])-1)//3)
            else:
                folds_counter[fold, classID+4*exercise_test] = folds_counter_original[fold, classID+4*exercise_test] + 1/(1+(len(df.loc[sampled_index, 'folds'])-1)//3)
        counter = (counter+1)%K

print(folds_counter_original)
print(folds_counter_original[:,0:7:2]+folds_counter_original[:,1:8:2])
print(folds_counter)
print(folds_counter[:,0:7:2]+folds_counter[:,1:8:2])

# Full print
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df)
    pass

# Save metadata file
df.to_csv(path/'metadata.csv', encoding='utf-8-sig', index=False)