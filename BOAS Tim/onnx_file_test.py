import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

import onnx
import onnxruntime as ort

import sys
import os

BOAS_folder_path = Path(__file__).resolve().parents[0]
sys.path.append(str(BOAS_folder_path))

from functions_and_classes import interpret_configuration, extract_audio_features, preprocess_one_file



# --------------
#   User input
# --------------

model_basename = 'NN_model22'
variable_part_of_name = 'noslice_aug_noet_2class_filter'
#data_folder = 'BOAS-mobilljud'
data_folder = 'Nya_Ljudfiler' # For test data

# sklearn specific settings:
save_features = True

slice_length = 30 # in seconds. Used as max duration if non-sliced data
slice_jump = 1

single_file = False
filename = 'hund1-post_converted.wav' # Overwritten if single_file = False´

use_all_files = False # overwritten by single_file = True
missing_et_info = False

print_individual_results = False # Mainly for sliced data



# ------------------
#   Main execution
# ------------------

# Build model filename
model_filename = f'{model_basename}_{variable_part_of_name}'

# Interpret configuration settings
use_sliced_data, _, use_exercise_test, before_and_after_et, use_2_classes, make_et_class, filter_signal, normalize_signal = interpret_configuration(variable_part_of_name)

# Change slice_length if data is sliced
if use_sliced_data:
    slice_length = 6
slice_settings = [use_sliced_data, slice_length, slice_jump]

# Define the class IDs
if use_2_classes:
    classes = [0,1]
else:
    classes = [0,1,2,3]

# Determine if labels are available or not
if missing_et_info and make_et_class or single_file:
    no_labels_available = True
else: 
    no_labels_available = False

# Check if the model is PyTorch CNN or sklearn classifier
is_sklearn = False
if 'sklearn' in model_filename: # Is sklearn
    is_sklearn = True

    approach_folder_path = BOAS_folder_path/'sklearn_'

    # Prepare to read metadata file
    data_path = BOAS_folder_path/data_folder
    metadata_filename = f'metadata{'_missinget'*missing_et_info}.csv'
    metadata_path = data_path/metadata_filename

    # Prepare for feature extraction
    features_filename = f'feature_names_{variable_part_of_name}.csv'
    pkl_folder_name = f'inference_pkl_data_{variable_part_of_name}'

    # Gather normalization data
    model_data_path = approach_folder_path/'Models'/'Model data'
    features_mean, features_std = np.loadtxt(model_data_path/('model_data_'+model_filename+'.csv'))

    # Collect/Extract features
    if single_file:
        audio_features = extract_audio_features(data_path, approach_folder_path, filename, variable_part_of_name, slice_settings=slice_settings, do_inference=True, single_file=single_file, save_features=save_features)
    else:
        audio_features = extract_audio_features(data_path, approach_folder_path, metadata_path, variable_part_of_name, slice_settings=slice_settings, do_inference=True, single_file=single_file, save_features=save_features)

    feature_matrix = np.vstack([audio.features for audio in audio_features])
    filenames = np.array([audio.filename for audio in audio_features])

    # Normalize feature matrix
    feature_matrix = (feature_matrix-features_mean)/features_std

    print(f'Files: {np.shape(feature_matrix)[0]}, Features per file: {np.shape(feature_matrix)[1]}')

else: # Is PyTorch
    
    approach_folder_path = BOAS_folder_path/'pytorch_'

# Load metadata
if single_file:

    df = pd.DataFrame.from_dict({'file_name': [filename], 'classID': [0]})

else:

    # Read metadata file
    data_path = BOAS_folder_path/data_folder 
    metadata_filename = 'metadata.csv'
    if missing_et_info: metadata_filename = 'metadata_missinget.csv'
    metadata_path = data_path/metadata_filename
    df = pd.read_csv(metadata_path)

    # Select audio files with 'exercise_test' = True or False depending on user input
    if not missing_et_info: 
        if not before_and_after_et:
            df = df[df['exercise_test']==use_exercise_test].reset_index(drop=True)
        elif make_et_class:
            df = df[[type(et) == type(True) for et in df['exercise_test']]]
    
    # Change classID to 0&1=0 and 2&3=1
    df_original_class = df.copy()
    if make_et_class and before_and_after_et:
        df['classID'] = np.int64(df['exercise_test'])
    elif use_2_classes: 
        df['classID'] = np.int64(df['classID']>1)

    # Find filenames of relevant files
    if not use_all_files:
        try:
            df['fold'] = [int(folds[0]) for folds in df['folds']]
        except:
            pass

    df = df.reset_index(drop=True)

# Load all models with the given name
all_model_files = os.listdir(approach_folder_path/'Models')
select_model_files = [filename.replace('.onnx','') for filename in all_model_files if model_filename in filename and '.onnx' in filename]

# Loop over all files
n_files = len(df)
for i_file in range(n_files):
    
    filename = df['file_name'][i_file]

    print(f'\n{filename}:\n')

    # Prepare data
    if is_sklearn: # Is sklearn

        # Prepare inference data
        select_file = [filename in fname for fname in filenames]
        X_test = feature_matrix[select_file]

        n_test = np.shape(X_test)[0]

    else: # Is PyTorch

        # Generate spectrograms as input to the model from the audio file
        file_path = BOAS_folder_path/data_folder/(filename)
        spectrogram_list = preprocess_one_file(file_path, channel=1, split_options=[use_sliced_data, slice_length, slice_jump], filter_signal=filter_signal)

        n_test = np.shape(spectrogram_list)[0]
    
    n_models = len(select_model_files)

    # Run through all selected models
    avg_prediction_list = np.zeros(n_models)
    avg_positive_confidence_list = np.zeros(n_models)
    for i_model, model_filename_i in enumerate(select_model_files):

        fold = np.int32(model_filename_i[-1])

        # Prepare model path for inference
        onxx_path = approach_folder_path/'Models'/(model_filename_i+'.onnx')

        if not is_sklearn: # Is PyTorch

            # Load and check consisteny of model
            onnx_model = onnx.load(onxx_path)
            onnx.checker.check_model(onnx_model, full_check=True)

        # Begin inference
        ort_session = ort.InferenceSession(onxx_path)
        input_name = ort_session.get_inputs()[0].name

        # Loop over all spectrograms, ie. all slices of the data file. Only one pass if not using slices
        prediction_list = np.zeros(n_test)
        positive_confidence_list = np.zeros(n_test)
        for i in range(n_test):
            
            if is_sklearn: # Is sklearn

                # Get test data
                x = X_test[i,:].reshape(1, -1)
                
                # Predict outputs
                outputs = ort_session.run(None, {input_name: x.astype(np.float32)})
                predicted, confidence = classes[outputs[0][0]], outputs[1][0][outputs[0][0]]

                # Grab BOAS positive probability for ROC
                y_pred_prob = outputs[1][0][1]

            else: # Is PyTorch

                # Get test data
                test_data = spectrogram_list[i]
                x = test_data.unsqueeze(0)

                # Normalize inputs
                x_m, x_s = x.mean(dim=(1, 2, 3), keepdim=True), x.std(dim=(1, 2, 3), keepdim=True)
                x = (x - x_m) / x_s
                
                # Predict outputs
                outputs = ort_session.run(None, {'input.1': x.numpy()})[0][0]
                predicted, confidence = classes[outputs.argmax(0)], (np.exp(outputs) / np.sum(np.exp(outputs), keepdims=True))[outputs.argmax(0)] # converting logits to probabilities

                # Grab class=positive probability for ROC
                y_pred_prob = (np.exp(outputs) / np.sum(np.exp(outputs), keepdims=True))[1] # converting logits to probabilities

            # Fill into lists
            prediction_list[i] = predicted
            positive_confidence_list[i] = y_pred_prob

            # Print single slice result
            if print_individual_results:
                print(f'Predicted: {'BOAS positive'*predicted + 'BOAS negative'*(not predicted)}, Confidence: {confidence}')

        # Evaluate single model results
        avg_prediction = int(np.sum(prediction_list) >= len(prediction_list)/2)
        avg_prediction_list[i_model] = avg_prediction
        avg_prediction_fraction = np.mean(prediction_list==avg_prediction)
        
        avg_positive_confidence = np.mean(positive_confidence_list)
        avg_positive_confidence_list[i_model] = avg_positive_confidence
        
        confidence_list = (not avg_prediction) + (avg_prediction-int(not avg_prediction))*positive_confidence_list
        confidence_list = positive_confidence_list if avg_prediction else 1-positive_confidence_list
        avg_confidence = np.mean(confidence_list)
        std_confidence = np.std(confidence_list)

        # Print single model results
        if n_test > 1:
            print(f'Model trained without fold {fold}: Majority Prediction: {'BOAS positive'*avg_prediction + 'BOAS negative'*(not avg_prediction)}, Fraction: {avg_prediction_fraction:.3f}, Average Confidence: {avg_confidence:.3f}, Standard Deviation: {std_confidence:.3f}')
        else:
            print(f'Model trained without fold {fold}: Predicted: {'BOAS positive'*avg_prediction + 'BOAS negative'*(not avg_prediction)}, Confidence: {avg_confidence:.3f}')
    
    # Evaluate average model results
    kfold_vote_prediction = int(np.sum(avg_prediction_list) >= len(avg_prediction_list)/2)
    kfold_vote_fraction = np.mean(avg_prediction_list==kfold_vote_prediction)
    
    kfold_avg_positive_confidence = np.mean(avg_positive_confidence_list)
    kfold_avg_prediction = int(kfold_avg_positive_confidence > 0.5)
    avg_confidence_list = avg_positive_confidence_list if kfold_avg_prediction else 1-avg_positive_confidence_list
    
    kfold_avg_confidence = np.mean(avg_confidence_list)
    kfold_std_confidence = np.std(avg_confidence_list)
    
    # Print average model results
    print(f'{n_models} Fold Vote Prediction:    {'BOAS positive'*kfold_vote_prediction + 'BOAS negative'*(not kfold_vote_prediction)}, Vote Fraction: {kfold_vote_fraction:.3f}')
    print(f'{n_models} Fold Average Prediction: {'BOAS positive'*kfold_avg_prediction + 'BOAS negative'*(not kfold_avg_prediction)}, Average Confidence: {kfold_avg_confidence:.3f}, Standard Deviation: {kfold_std_confidence:.3f}')