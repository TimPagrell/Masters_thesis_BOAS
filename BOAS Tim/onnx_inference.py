import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
from sklearn import metrics

import onnx
import onnxruntime as ort

import sys
import os

BOAS_folder_path = Path(__file__).resolve().parents[0]
sys.path.append(str(BOAS_folder_path))

from functions_and_classes import interpret_configuration, extract_audio_features, preprocess_one_file, ROC_analysis



# --------------
#   User input
# --------------

#model_basename = 'sklearn_model_2'
#variable_part_of_name = 'slice_aug_2_noet_2class_filter_noNorm'
model_basename = 'NN_model22_psts'
variable_part_of_name = 'noslice_aug_et_2class_filter'
data_folder = 'BOAS-mobilljud'
data_folder = 'Nya_Ljudfiler' # For test data

# sklearn specific settings:
save_features = True

slice_length = 30 # in seconds. Used as max duration if non-sliced data
slice_jump = 1 # in seconds

single_file = False
filename = 'hund1-post_converted.wav' # Overwritten if single_file = False´

use_all_files = False # Use all files instead of fold-wise. Overwritten by single_file = True
missing_et_info = False # If we do not know if the data is before or after et

do_plot = False # ROC, confusion matrix, slice accuracy chart (if sliced data)
custom_plot_title = '' # custom title of the sliced data
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
    
    # Gather normalization data
    model_data_path = approach_folder_path/'Models'/'Model data'
    features_mean, features_std = np.loadtxt(model_data_path/('model_data_'+model_filename+'.csv'))

    # Collect/Extract features
    if single_file:
        audio_features = extract_audio_features(data_path, approach_folder_path, filename, variable_part_of_name, slice_settings=slice_settings, do_inference=True, single_file=single_file, save_features=save_features)
    else:
        audio_features = extract_audio_features(data_path, approach_folder_path, metadata_path, variable_part_of_name, slice_settings=slice_settings, do_inference=True, single_file=single_file, save_features=save_features)

    feature_matrix = np.vstack([audio.features for audio in audio_features])
    labels = np.array([audio.label for audio in audio_features])
    folds = np.array([audio.fold for audio in audio_features])
    filenames = np.array([audio.filename for audio in audio_features])

    # Normalize feature matrix
    feature_matrix = (feature_matrix-features_mean)/features_std

    print(f'Files: {np.shape(feature_matrix)[0]}, Features per file: {np.shape(feature_matrix)[1]}')

else: # Is PyTorch
    
    approach_folder_path = BOAS_folder_path/'pytorch_'

# Find all models with the given name
all_model_files = os.listdir(approach_folder_path/'Models')
select_model_files = [filename.replace('.onnx','') for filename in all_model_files if model_filename in filename and '.onnx' in filename]
n_models = len(select_model_files)

# Prepare model averaged result lists
modelavg_accuracy_list = np.zeros(n_models)
n_items_model_list = np.zeros(n_models)
all_fileavg_confidence_of_true_label_list = []
all_models_y_pred_prob_list = []
all_models_labels = []
all_models_predictions = []

# Run through all selected models
for i_model, model_filename_i in enumerate(select_model_files):

    print(f'\n{model_filename_i}:\n')

    fold = np.int32(model_filename_i[-1])

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

        # Find fold-relevant files
        if not use_all_files:
            try:
                df['fold'] = [int(folds[0]) for folds in df['folds']]
            except:
                pass

            try:
                df = df[df['fold']==fold]
            except:
                pass

        df = df.reset_index(drop=True)

    # Rename 'slice_file_name' to 'file_name', for consistency
    if 'slice_file_name' in df.columns:
        df['file_name'] = df['slice_file_name']
    
    # If required by missing_et_info = True and 'etclass' model, get a separate account on the BOAS class
    if no_labels_available:
        df_classes = pd.read_csv(metadata_path)

    # Prepare model path for inference
    onxx_path = approach_folder_path/'Models'/(model_filename_i+'.onnx')

    if not is_sklearn: # Is PyTorch

        # Load and check consisteny of model
        onnx_model = onnx.load(onxx_path)
        onnx.checker.check_model(onnx_model, full_check=True)

    # Begin inference
    ort_session = ort.InferenceSession(onxx_path)
    input_name = ort_session.get_inputs()[0].name

    # Prepare lists for plotting
    if do_plot:
        filename_list = []
        prediction_accuracy_list = []
        original_class_list = []

    # Prepare result lists
    result_list = []
    y_test_list = []
    y_pred_prob_list = []
    all_labels = []
    all_predictions = []
    n_files = len(df)
    fileavg_confidence_of_true_label_list = np.zeros(n_files)

    # Loop over all files
    for i_file in range(n_files):
        
        filename = df['file_name'][i_file]
        true_class = np.int32(df['classID'])[i_file]

        if not no_labels_available and do_plot:
            filename_list.append(filename.replace('.wav',''))
            original_class_list.append(f'BOAS {np.int32(df_original_class['classID'])[i_file]}')

        if is_sklearn: # Is sklearn

            # Prepare inference data
            select_file = [filename in fname for fname in filenames]
            X_test = feature_matrix[select_file]
            y_test = labels[select_file]

            n_test = np.shape(y_test)[0]

        else: # Is PyTorch

            # Generate spectrograms as input to the model from the audio file
            file_path = BOAS_folder_path/data_folder/(filename)
            spectrogram_list = preprocess_one_file(file_path, channel=1, split_options=[use_sliced_data, slice_length, slice_jump], filter_signal=filter_signal)

            n_test = np.shape(spectrogram_list)[0]

        label_list = np.zeros(n_test)
        prediction_list = np.zeros(n_test)
        confidence_of_true_label_list = np.zeros(n_test)
        for i in range(n_test):
            
            if is_sklearn: # Is sklearn
                # Get test data
                x, y = X_test[i,:].reshape(1, -1), y_test[i]
                if no_labels_available:
                    y = df_classes['classID'][i]
                
                # Predict outputs
                outputs = ort_session.run(None, {input_name: x.astype(np.float32)})
                predicted, actual, confidence = classes[outputs[0][0]], classes[y], outputs[1][0][outputs[0][0]]
                
                # Grab BOAS positive probability for ROC
                y_pred_prob = outputs[1][0][1]

            else: # Is PyTorch

                # Get test data
                test_data = spectrogram_list[i]
                x, y = test_data.unsqueeze(0), true_class
                if no_labels_available and not single_file:
                    y = df_classes['classID'][i]

                # Normalize inputs
                x_m, x_s = x.mean(dim=(1, 2, 3), keepdim=True), x.std(dim=(1, 2, 3), keepdim=True)
                x = (x - x_m) / x_s
                
                # Predict outputs
                outputs = ort_session.run(None, {'input.1': x.numpy()})[0][0]
                predicted, actual, confidence = classes[outputs.argmax(0)], classes[y], (np.exp(outputs) / np.sum(np.exp(outputs), keepdims=True))[outputs.argmax(0)] # converting logits to probabilities

                # Grab class=positive probability for ROC
                y_pred_prob = (np.exp(outputs) / np.sum(np.exp(outputs), keepdims=True))[1] # converting logits to probabilities
            
            # Confidence of the actual label
            confidence_of_true_label = confidence if predicted==actual else 1-confidence

            # Fill into lists
            label_list[i] = actual
            prediction_list[i] = predicted
            y_test_list.append(y)
            y_pred_prob_list.append(y_pred_prob)

            all_models_labels.append(actual)
            all_models_predictions.append(predicted)
            all_models_y_pred_prob_list.append(y_pred_prob)

            confidence_of_true_label_list[i] = confidence_of_true_label
            
            # Print single result
            if print_individual_results:
                print(f'Predicted: {predicted}, Actual: {actual}')

        if do_plot: 
            all_labels.append(label_list)
            all_predictions.append(prediction_list)
            prediction_accuracy_list.append(np.int32(label_list==prediction_list))

        # Calculate accuracy
        accuracy = np.sum(label_list==prediction_list)/n_test
        result_list.append([accuracy,n_test])

        # Average confidence of true labels
        fileavg_confidence_of_true_label = np.mean(confidence_of_true_label_list)
        fileavg_confidence_of_true_label_list[i_file] = fileavg_confidence_of_true_label
        all_fileavg_confidence_of_true_label_list.append(fileavg_confidence_of_true_label)

        # Print full result
        if use_sliced_data:

            print(f'\n{filename}: True Label: {true_class}, Predictions:')
            print(prediction_list)
            print(f'Accuracy: {accuracy:.2f}, Average Confidence of True Label: {fileavg_confidence_of_true_label:.3f}, Total Items: {n_test}')

        else:

            name_max_len = np.max([len(filename) for filename in df['file_name']])
            if no_labels_available and not single_file:
                print(f'{f'{filename}:'.ljust(name_max_len+1)} Predicted: {'Post ET,'*predicted + 'Pre ET,'*(not predicted):<8} Confidence: {f'{confidence:.3f},':<6} BOAS Grade: {actual}')
            elif single_file:
                print(f'{f'{filename}:'.ljust(name_max_len+1)} Predicted: {'BOAS positive,'*predicted + 'BOAS negative,'*(not predicted)} Confidence: {f'{confidence:.3f},':<6}')
            else:
                print(f'{f'{filename}:'.ljust(name_max_len+1)} {f'{'in'*(predicted!=actual)}correct,':<10} Predicted: {predicted}, Actual: {actual}, Confidence: {confidence:.3f}')

    if not no_labels_available:

        # Calculate averaged results
        result_list = np.array(result_list)
        modelavg_accuracy = np.mean(result_list[:,0])
        modelavg_confidence_of_true_label = np.mean(fileavg_confidence_of_true_label_list)
        n_items_model = int(np.sum(result_list[:,1]))

        # ROC analysis:
        roc_auc = ROC_analysis(y_test_list, y_pred_prob_list, do_plot=do_plot)

        # Append to lists
        modelavg_accuracy_list[i_model] = modelavg_accuracy
        n_items_model_list[i_model] = n_items_model

        # Print averaged results
        print(f'\nFile-Averaged Accuracy: {modelavg_accuracy:.3f}, File-Averaged Confidence of True Label: {modelavg_confidence_of_true_label:.3f}, ROC-AUC: {roc_auc:.3f}, Total Items: {n_items_model}\n')

        # Visualize with imshow (if sliced data), confusion matrix
        if do_plot:

            # Plot confusion matrix
            confusion_matrix = metrics.confusion_matrix(np.concatenate(all_labels), np.concatenate(all_predictions))
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = classes)
            cm_display.plot()
            plt.show()  

            if use_sliced_data:
            
                # Generate image for imshow
                max_length = np.max([len(prediction) for prediction in prediction_accuracy_list])
                image = np.zeros([len(prediction_accuracy_list), max_length]) + 0.5
                for i_row, row in enumerate(prediction_accuracy_list):
                    image[i_row,:len(row)] = row

                fig, ax = plt.subplots(figsize=(16,5))

                colours = ['red', 'white', 'green']
                bounds = [-0.5, 0.25, 0.75, 1.5]
                cmap = ListedColormap(colours)
                norm = BoundaryNorm(bounds, cmap.N)

                ax.imshow(image, cmap=cmap, norm=norm)

                ax.set_xticks(np.arange(-0.5, image.shape[1], 1), minor=True)
                ax.set_yticks(np.arange(-0.5, image.shape[0], 1), minor=True)
                ax.set_xticks([], labels=[])
                ax.set_yticks(np.arange(image.shape[0]), labels=filename_list)
                ax.set_xlim(-0.5, image.shape[1] - 0.5)

                ax_right_y = ax.twinx()
                ax_right_y.imshow(image, alpha=0, cmap=cmap, norm=norm)

                ax_right_y.set_xticks([], labels=[])
                ax_right_y.set_yticks(np.arange(image.shape[0]), labels=original_class_list)
                ax_right_y.set_xlim(-0.5, image.shape[1] - 0.5)

                ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
                
                if len(custom_plot_title)>0:
                    ax.set_title(custom_plot_title)
                elif single_file:
                    ax.set_title(f'Full File Inference ({(not use_exercise_test)*'no'}et Model): {filename}')
                elif use_sliced_data:
                    ax.set_title(f'Full Fold Inference ({(not use_exercise_test)*'no'}et Model): Fold {fold}')
                else:
                    ax.set_title(f'Full Set Inference ({(not use_exercise_test)*'no'}et Model)')

                plt.show()

                # Save figure
                figures_path = BOAS_folder_path/'Figures'
                if missing_et_info: save_plot_filename = model_filename.replace('NN_model','Missing_et_info')+'.png'
                else: save_plot_filename = model_filename.replace('NN_model','Full_file_test')+'.png'
                fig.savefig(figures_path/save_plot_filename)

if not no_labels_available:

    # Plot confusion matrix for combined models
    if do_plot:
        confusion_matrix = metrics.confusion_matrix(all_models_labels, all_models_predictions)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = classes)
        cm_display.plot()
        plt.show()  
    
    # ROC on all models results
    all_roc_auc = ROC_analysis(all_models_labels, all_models_y_pred_prob_list, do_plot=do_plot)

    # Print averaged results
    mean_accuracy = np.mean(modelavg_accuracy_list)
    mean_confidence_of_true_label = np.mean(all_fileavg_confidence_of_true_label_list)
    std_confidence_of_true_label = np.std(all_fileavg_confidence_of_true_label_list)
    total_items = int(np.sum(n_items_model_list))
    print(f'\nAverage Across All Models:\nAccuracy: {mean_accuracy:.3f}, ROC-AUC: {all_roc_auc:.3f}, Confidence of True Label: {mean_confidence_of_true_label:.3f}, Standard Deviation: {std_confidence_of_true_label:.3f}, Total Items: {total_items}\n')     