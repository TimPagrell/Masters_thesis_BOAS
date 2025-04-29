import pandas as pd
from pathlib import Path
import numpy as np
import copy

import torch
from torch.utils.data import DataLoader
from torchinfo import summary
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

import sys
import os

script_path = Path(__file__).resolve().parents[0]
BOAS_folder_path = Path(__file__).resolve().parents[1]
sys.path.append(str(BOAS_folder_path))

from functions_and_classes import AudioUtil, SoundDS, training, inference



# --------------
#   User input
# --------------

#device = 'cpu' # Uncomment to use cpu instead of gpu (likely slower)

data_folder = 'BOAS-mobilljud'
max_runs = 3 # The number of runs that the model is trained, from which the model with the best validation results are automatically chosen.

use_2_classes = True # Simplifies the model from the four BOAS grades to only 2, BOAS negative (0&1) and BOAS positive (2&3).
use_exercise_test = True # Makes it so the model only trains on audio from after exercise test.
filter_signal = True # If the adaptive high-pass filter should be used.
use_sliced_data = True # If sliced data should be used.
use_augmented_data = False # If offline-augmented data should be used.
augmented_folder_suffix = '' # If a specific augmentation should be used, defined when augmenting the data.
val_fold = 1 # The fold that should not be included in training (K fold cross-validation).

separate_augments = False # Used iff use_augmented_data = True. Use True for one augmented version per original data per epoch, False if all augmented data each epoch.

before_and_after_et = False # Overrides use_exercise_test. Makes it so the model trains on audio from both before and after exercise test, and any audio where that is uncertain.
hybrid_et = False # Used iff before_and_after_et = True. Makes it so each input contains two files, one from before et and the other from the same dog after et.
make_et_class = False # Used iff before_and_after_et = True, and overrides use_2_class. Makes it so the model now trains to determine if the audio is from before or after exercise test. Does not work with hybrid_et = True.

# Settings that depend on above inputs
if use_sliced_data:
    duration = 6 # in seconds
    num_epochs = 100
    batch_size = 32
    learning_rate = 1e-3
    from pytorch_models import AudioClassifierSlice as AudioClassifier
    model_suffix = '3' # '3' was the model iteration identifier. Holds no significance

    if use_augmented_data:
        model_suffix += augmented_folder_suffix
        if separate_augments:
            num_epochs = 300
            pass
        else:
            num_epochs = 50
            batch_size = 64

else:
    duration = 30 # in seconds
    num_epochs = 300
    batch_size = 6
    learning_rate = 1e-3
    if use_exercise_test:
        from pytorch_models import AudioClassifierNoSlice as AudioClassifier
        model_suffix = '22' # '22' (not twenty two..) was the model iteration identifier. Holds no significance
    else:
        from pytorch_models import AudioClassifierNoSlice as AudioClassifier
        model_suffix = '22'

    if use_augmented_data:
        model_suffix += augmented_folder_suffix
        if separate_augments:
            num_epochs = 500
            pass
        else:
            num_epochs = 100
            batch_size = 16



# ------------------
#   Main execution
# ------------------

val_folds = [val_fold]

# Prepare names for saving
slice_part_of_name = 'no'*(not use_sliced_data)+'slice'
aug_part_of_name = 'no'*(not use_augmented_data)+'aug'
et_part_of_name = f'{('no'*(not use_exercise_test))*(not before_and_after_et) + 'both'*(before_and_after_et and not hybrid_et) + 'hybrid'*(before_and_after_et and hybrid_et)}et'
class_part_of_name = f'{2+2*(not use_2_classes)}class'*(not (make_et_class and before_and_after_et)) + 'etclass'*(make_et_class and before_and_after_et)
filter_part_of_name = 'no'*(not filter_signal)+'filter'
fold_part_of_name = f'fold{val_folds[0]}'
variable_part_of_name = f'{slice_part_of_name}_{aug_part_of_name}_{et_part_of_name}_{class_part_of_name}_{filter_part_of_name}_{fold_part_of_name}'

save_name = f'NN_model{model_suffix}_{variable_part_of_name}'
save_training_plot_filename = f'NN_model{model_suffix}_training_{variable_part_of_name}'

# Read metadata files
if use_augmented_data and use_sliced_data:
    train_data_path = BOAS_folder_path/(f'{data_folder}-augmented{augmented_folder_suffix}-sliced')
    val_data_path = BOAS_folder_path/(f'{data_folder}-sliced')

elif use_augmented_data:
    train_data_path = BOAS_folder_path/(f'{data_folder}-augmented{augmented_folder_suffix}')
    val_data_path = BOAS_folder_path/data_folder

elif use_sliced_data:
    train_data_path = BOAS_folder_path/(f'{data_folder}-sliced')
    val_data_path = train_data_path

else: 
    train_data_path = BOAS_folder_path/data_folder
    val_data_path = train_data_path

df_train = pd.read_csv(train_data_path/f'metadata{'_hybrid'*(before_and_after_et and hybrid_et)}.csv') 
df_val = pd.read_csv(val_data_path/f'metadata{'_hybrid'*(before_and_after_et and hybrid_et)}.csv')

# Select audio files with 'exercise_test' = True or False depending on user input
if not before_and_after_et:
    df_train = df_train[df_train['exercise_test']==use_exercise_test].reset_index(drop=True)
    df_val = df_val[df_val['exercise_test']==use_exercise_test].reset_index(drop=True)

elif make_et_class:
    df_train = df_train[[type(et) == type(True) for et in df_train['exercise_test']]]
    df_val = df_val[[type(et) == type(True) for et in df_val['exercise_test']]]

# Change classID to 0&1=0 and 2&3=1
if make_et_class and before_and_after_et:
    df_train['classID'] = np.int64(df_train['exercise_test'])
    df_val['classID'] = np.int64(df_val['exercise_test'])

elif use_2_classes:
    df_train['classID'] = np.int64(df_train['classID']>1)
    df_val['classID'] = np.int64(df_val['classID']>1)

# Rename 'slice_file_name' to 'file_name', for consistency
if 'slice_file_name' in df_train.columns:
    df_train['file_name'] = df_train['slice_file_name']
    df_val['file_name'] = df_val['slice_file_name']

# Construct relative file paths
if use_augmented_data and use_sliced_data:
    df_train['relative_path'] = '/augment' + df_train['augment_number'].astype(str) + '/fold' + df_train['fold'].astype(str) + '/'# + df_train['slice_file_name'].astype(str)
    df_val['relative_path'] = '/fold' + df_val['fold'].astype(str) + '/'# + df_val['slice_file_name'].astype(str)

elif use_augmented_data:
    df_train['relative_path'] = '/augment' + df_train['augment_number'].astype(str) + '/'# + df_train['file_name'].astype(str)
    df_val['relative_path'] = '/'# + df_val['file_name'].astype(str)

    # Only take into account the first fold per file, to avoid overlap
    df_train['fold'] = [int(folds[0]) for folds in df_train['folds']]
    df_val['fold'] = [int(folds[0]) for folds in df_val['folds']]

elif use_sliced_data:
    df_train['relative_path'] = '/fold' + df_train['fold'].astype(str) + '/'# + df_train['slice_file_name'].astype(str)
    df_val['relative_path'] = df_train['relative_path']

else: 
    df_train['relative_path'] = '/'# + df_train['file_name'].astype(str)
    df_val['relative_path'] = df_train['relative_path']

    # Only take into account the first fold per file, to avoid overlap
    df_train['fold'] = [int(folds[0]) for folds in df_train['folds'].astype(str)]
    df_val['fold'] = [int(folds[0]) for folds in df_val['folds'].astype(str)]

# Get sample rate from one file, to use for all
if hybrid_et and before_and_after_et:
    _, sr = AudioUtil.open(str(train_data_path)+df_train['relative_path'][0]+df_train['file_name_pre'][0])
else:
    _, sr = AudioUtil.open(str(train_data_path)+df_train['relative_path'][0]+df_train['file_name'][0])

# Prepare training (both for training and validation) and validation data
train_folds = (np.arange(5)+1)[[fold not in val_folds for fold in np.arange(5)+1]]

train_mask = df_train['fold'].isin(train_folds)
train_df = df_train[train_mask].sample(frac=1).reset_index(drop=True)

train_val_mask = df_val['fold'].isin(train_folds)
train_val_df = df_val[train_val_mask].sample(frac=1).reset_index(drop=True)

val_mask = df_val['fold'].isin(val_folds)
val_df = df_val[val_mask].sample(frac=1).reset_index(drop=True)

train_ds = SoundDS(train_val_df, val_data_path, duration=duration, sr=sr, hybrid_et=hybrid_et, filter_signal=filter_signal)
val_ds = SoundDS(val_df, val_data_path, duration=duration, sr=sr, hybrid_et=hybrid_et, filter_signal=filter_signal)

# Create training (for validation) and validation data loaders
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)#, num_workers=4)
val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)#, num_workers=4)

train_new_model = True
highest_F1_score = -1
bestModel = None
fig_bestModel = None
i_run = 0
while train_new_model:
    i_run += 1

    # Create model
    num_classes = 2+2*(not use_2_classes)
    myModel = AudioClassifier(num_classes=num_classes, hybrid_et=hybrid_et)
    myModel = myModel.to(device)

    # Print model summary
    input_shape = list(np.shape(val_ds.__getitem__(0)[0].unsqueeze(0)))
    if i_run == 1:
        print('Model summary :')
        summary(myModel, input_shape)

        # Check that it is on Cuda (or not)
        print(f'Current device: {next(myModel.parameters()).device}')

    print(f'\nRun {i_run}')

    # Train model
    myModel.train()
    myModel, fig, val_loss_list = training(myModel, train_df, train_data_path, duration, sr, batch_size, val_dl, num_epochs, device, learning_rate, return_best_model=True, use_sliced_data=use_sliced_data, 
                                            use_augmented_data=use_augmented_data, do_plot=True, separate_augments=separate_augments, hybrid_et=hybrid_et, filter_signal=filter_signal)

    # Validation of the model
    myModel.eval()
    val_acc, val_loss, F1_score, roc_auc = inference(myModel, val_dl, device)

    # Compare if the trained model is the best one yet, if so save it for now
    if F1_score > highest_F1_score:
        highest_F1_score = F1_score
        highest_roc_auc = 0

    if F1_score == highest_F1_score:

        if roc_auc > highest_roc_auc:
            highest_roc_auc = roc_auc
            lowest_loss = np.inf

        if roc_auc == highest_roc_auc:

            if val_loss < lowest_loss:
                lowest_loss = val_loss
                bestModel = copy.deepcopy(myModel)
                fig_bestModel = copy.deepcopy(fig)

    # Determine if the run should continue or not
    if i_run >= max_runs:
        train_new_model = False
    else:
        train_new_model = True

# Final inference checks and roc with plots
print('\nFinal Inference')
bestModel.eval()
inference(bestModel, val_dl, device, do_plot=True)
inference(bestModel, train_dl, device, do_plot=True)

# Save model based on user input
print(f'\nSave name: {save_name}')
user_input = input('Do you want to save the model? (yes/no): ')
if not user_input.lower() in ['yes', 'y']:
    user_input = input('Are you sure? The model will disappear permanently if yes. (yes/no): ')
    if not user_input.lower() in ['no', 'n', 'save']:
        print('Did not save')
        quit()

# Save training plot
if fig_bestModel != None:
    figures_path = BOAS_folder_path/'Figures'
    Path(figures_path).mkdir(parents=True, exist_ok=True)
    save_training_plot_filename += '.png'
    fig_bestModel.savefig(figures_path/save_training_plot_filename)

# Turn model on evaluation mode before exporting
bestModel.eval()

# Save model
save_path = script_path/'Models'
Path(save_path).mkdir(parents=True, exist_ok=True)

# as torch file
torch.save(bestModel.state_dict(), save_path/save_name)

# as onnx file
torch.onnx.export(
    bestModel,
    val_ds.__getitem__(0)[0].unsqueeze(0).to(device),
    save_path/(save_name+'.onnx'),
    export_params=True,
    do_constant_folding=True
)

print('Did save')