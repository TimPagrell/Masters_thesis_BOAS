import pandas as pd
from pathlib import Path
import pickle
import numpy as np

import sklearn.svm as svm
import sklearn.ensemble as en #import RandomForestClassifier
import sklearn.linear_model as lm # import LogisticRegression, RidgeClassifier, SGDClassifier, PassiveAggressiveClassifier
import sklearn.neighbors as nb
from sklearn.neural_network import MLPClassifier

import sys
import os

script_path = Path(__file__).resolve().parents[0]
BOAS_folder_path = Path(__file__).resolve().parents[1]
sys.path.append(str(BOAS_folder_path)) 

from functions_and_classes import interpret_configuration, FeaturesModel, extract_audio_features



# --------------
#   User input
# --------------

data_folder = 'BOAS-mobilljud'
variable_part_of_name = 'slice_aug_2_et_2class_nofilter_noNorm'
save_features = True # For faster reruns

model_suffix = ''
augmented_folder_suffix = ''
online_augments = 0

""" model_cfg = dict(
    model=en.RandomForestClassifier(
        n_jobs=-1,
        class_weight="balanced",
        n_estimators=200,
        bootstrap=True
    ),
) """

model_cfg = dict(
    model=lm.LogisticRegression(
        max_iter=500,
        fit_intercept=True,
        tol=1e-5,
        solver='lbfgs'
    ),
)

""" model_cfg = dict(
    model=lm.RidgeClassifier(
        max_iter=500,
        fit_intercept=True,
        tol=1e-5,
        solver='auto'
    ),
) """

""" model_cfg = dict(
    model=lm.SGDClassifier(
        max_iter=500,
        fit_intercept=True,
        tol=1e-5,
        loss='log_loss'
    ),
) """

""" model_cfg = dict(
    model=lm.PassiveAggressiveClassifier(
        max_iter=500,
        fit_intercept=True,
        tol=1e-5,
        loss='hinge'
    ),
) """

""" model_cfg = dict(
    model=en.BaggingClassifier(
        n_jobs=-1,
        n_estimators=500,
        bootstrap=True
    ),
) """

""" model_cfg = dict(
    model=en.ExtraTreesClassifier(
        n_jobs=-1,
        class_weight="balanced",
        n_estimators=100,
        bootstrap=True
    ),
) """

""" model_cfg = dict(
    model=en.GradientBoostingClassifier(
        n_estimators=100
    ),
) """

""" estimators = [
    ('svr', svm.SVC(
        kernel='sigmoid',
        tol=1e-5,
        probability=True
    ))
]
model_cfg = dict(
    model=en.StackingClassifier(
        passthrough=False,
        estimators=estimators,
        n_jobs=-1,
        final_estimator=lm.LogisticRegression()
    ), 
) """

""" from sklearn.naive_bayes import GaussianNB
estimators = [
    ('lr', lm.LogisticRegression(
        max_iter=500,
        fit_intercept=True,
        tol=1e-5,
        solver='lbfgs'
    )),
    ('rfc', en.RandomForestClassifier(
        n_jobs=-1,
        class_weight="balanced",
        n_estimators=500,
        bootstrap=True
    )),
    ('svc', svm.NuSVC(
        nu=0.5,
        kernel='sigmoid',
        tol=1e-5,
        probability=True,
        random_state=42
    ))
]
model_cfg = dict(
    model=en.VotingClassifier(
        estimators=estimators,
        n_jobs=-1,
        voting='soft',
        flatten_transform=False
    ),
) """

""" model_cfg = dict(
    model=svm.SVC(
        kernel='sigmoid',
        tol=1e-5,
        probability=True,
        random_state=42
    ),
) """

""" model_cfg = dict(
    model=svm.NuSVC(
        nu=0.5,
        kernel='sigmoid',
        tol=1e-5,
        probability=True,
        random_state=42
    ),
) """

""" model_cfg = dict(
    model=nb.KNeighborsClassifier(
        n_neighbors = 5,
    ),
) """

""" hidden_layer_sizes = (16,16)
model_cfg = dict(
    model=MLPClassifier(
        hidden_layer_sizes, 
        solver='adam', 
        activation='tanh',
        tol=1e-4, 
        n_iter_no_change=100, 
        max_iter=1000,
        alpha=0.0, 
        momentum=0.9, 
        learning_rate_init=0.1
    ),
) """



# ------------------
#   Main execution
# ------------------

model_suffix += augmented_folder_suffix

# Build model filename
model_suffix += (not not online_augments)*f'_{online_augments}'
model_filename = f'sklearn_model{model_suffix}_{variable_part_of_name}'

# Interpret configuration settings
use_sliced_data, use_augmented_data, _, _, _, _, _, _ = interpret_configuration(variable_part_of_name)

if use_augmented_data:
    data_folder += f'-augmented{augmented_folder_suffix}'

if use_sliced_data:
    data_folder += '-sliced'

# Prepare paths to extract features
data_path = BOAS_folder_path/data_folder
metadata_path = data_path/'metadata.csv'

# Extract features
audio_features = extract_audio_features(data_path, script_path, metadata_path, variable_part_of_name, suffix=augmented_folder_suffix, online_augments=online_augments, save_features=save_features)

feature_matrix = np.vstack([audio.features for audio in audio_features])
labels = np.array([audio.label for audio in audio_features])
folds = np.array([audio.fold for audio in audio_features])

# Normalize data (need to save the mean and std for inference)
features_mean = np.mean(feature_matrix, axis=0)
features_std = np.std(feature_matrix, axis=0)
feature_matrix = (feature_matrix-features_mean)/features_std

# Save the mean and std into Model data
model_data_path = script_path/'Models'/'Model data'
Path(model_data_path).mkdir(parents=True, exist_ok=True)
np.savetxt(model_data_path/('model_data_'+model_filename+'.csv'), [features_mean, features_std])

print(f'Files: {np.shape(feature_matrix)[0]}, Features per file: {np.shape(feature_matrix)[1]}')

# Train model
model = FeaturesModel(feature_matrix, labels, folds, model_cfg)
val_accuracy = model.train_kfold(do_save_each_model=True, model_filename=model_filename, folder_path=script_path)

print(val_accuracy)
print(f'Mean validation accuracy: {np.mean(val_accuracy):.3f}')