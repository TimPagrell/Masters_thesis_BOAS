from pathlib import Path
import os
import numpy as np
import librosa
from time import time
from sklearn import metrics
import pickle
import pandas as pd
import scipy
import copy

import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

import torch
from torchaudio import transforms, functional
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch import nn

from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

import opensmile
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)



# ---------------------------------------
#   Pytorch model functions and classes
# ---------------------------------------


# Collection of audio utilities
class AudioUtil():

    # ----------------------------
    # Load an audio file. Return the signal as a tensor and the sample rate
    # ----------------------------
    @staticmethod
    def open(audio_file):
        signal, sr = torchaudio.load(audio_file) # sr = "sample rate"
        return (signal, sr)
    
    # ----------------------------
    # Save an audio signal to a file. 
    # ----------------------------
    @staticmethod
    def save(audio, path, path_is_relative=True):
        signal, sr = audio
        if path_is_relative == True:
            path = Path.cwd()/path
            torchaudio.save(path, signal, sr)
        else: 
            torchaudio.save(path, signal, sr)
        return None

    # ----------------------------
    # Convert the given audio to the desired number of channels
    # ----------------------------
    @staticmethod
    def rechannel(audio, new_channel):
        signal, sr = audio

        if (signal.shape[0] == new_channel):
            # Nothing to do
            return audio

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resignal = signal[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resignal = torch.cat([signal, signal])

        return ((resignal, sr))

    # ----------------------------
    # Since Resample applies to a single channel, we resample one channel at a time
    # ----------------------------
    @staticmethod
    def resample(audio, newsr):
        signal, sr = audio

        if (sr == newsr):
            # Nothing to do
            return audio

        num_channels = signal.shape[0]
        # Resample first channel
        resignal = torchaudio.transforms.Resample(sr, newsr)(signal[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, newsr)(signal[1:,:])
            resignal = torch.cat([resignal, retwo])

        return ((resignal, newsr))

    # ----------------------------
    # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
    # ----------------------------
    @staticmethod
    def pad_trunc(audio, max_ms):
        signal, sr = audio
        num_rows, signal_len = signal.shape
        max_len = sr//1000 * max_ms
        
        if (signal_len > max_len):
            # Truncate the signal to the given length
            signal = signal[:,:max_len]

        elif (signal_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = np.random.randint(max_len - signal_len)
            pad_end_len = max_len - signal_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            signal = torch.cat((pad_begin, signal, pad_end), 1)

        return (signal, sr)
    
    # ----------------------------
    # Find noise level of audio signal
    # ----------------------------
    @staticmethod
    def find_noise_level(signal, sr, d=0.1):
        signal_length = len(signal[0])/sr
        n_points = np.int32(signal_length/d)
        max_list = np.zeros(n_points)
        time_list = np.zeros(n_points)
        for i in range(n_points-1):
            time_list[i] = (i+0.5)*d

            # Skip first second, as some devices needs time to "warm up"
            if i*d<1:
                max_list[i] = 0
                continue
            
            max_list[i] = np.abs(signal[0][np.int32(i*d*sr):np.int32((i+1)*d*sr)]).max()
        else:
            time_list[i+1] = (i+1.5)*d
            max_list[i+1] = np.abs(signal[0][np.int32((i+1)*d*sr):]).max()

        max_list[:np.int32(1/d)] = max_list[np.int32(1/d)]

        noise_level = np.min(max_list)

        return noise_level
    
    # ----------------------------
    # Normalize audio, given as (signal, sr), by noise level. 
    # ----------------------------
    @staticmethod
    def normalize_audio_by_noise_level(audio):
        signal, sr = audio

        noise_level = AudioUtil.find_noise_level(signal, sr, d=0.1)
        normalized_signal = signal/np.min(noise_level)
            
        return (normalized_signal, sr) 
    
    # ----------------------------
    # Normalize audio, given as (signal, sr), by the Root Mean Squared (RMS). 
    # ----------------------------
    @staticmethod
    def normalize_audio_by_RMS(audio):
        signal, sr = audio

        rms = np.sqrt((signal ** 2).mean(dim=1))
        rms = rms.reshape(rms.size(0), -1)

        normalized_signal = signal/rms
            
        return (normalized_signal, sr) 

    # ----------------------------
    # Shifts the signal to the left or right by some percent. Values at the end
    # are 'wrapped around' to the start of the transformed signal.
    # ----------------------------
    @staticmethod
    def time_shift(audio, shift_limit, randomize=True):
        signal, sr = audio
        _, signal_len = signal.shape
        if randomize:
            shift_amt = int(np.random.rand() * shift_limit * signal_len)
        else:
            shift_amt = int(shift_limit * signal_len)
        return (signal.roll(shift_amt), sr)
    
    # ----------------------------
    # Adds white noise to the signal, with strength corresponding to 
    # a noise factor and the standard deviation of the signal.
    # ----------------------------
    @staticmethod
    def add_white_noise(audio, noise_factor):
        signal, sr = audio
        noise = np.sqrt(0.1)*torch.randn(signal.size())
        noise_level = AudioUtil.find_noise_level(signal, sr)
        augmented_signal = signal + noise*noise_factor*noise_level
        return (augmented_signal, sr)
    
    # ----------------------------
    # Change the speed of the audio file by factor stretch_rate
    # without affecting the pitch. Use before pad/truncating.
    # stretch_rate > 1 means speed up the signal
    # stretch_rate < 1 means slow down the signal
    # ----------------------------
    @staticmethod
    def time_stretch(audio, stretch_rate):
        signal, sr = audio
        augmented_signal = librosa.effects.time_stretch(np.array(signal), rate=stretch_rate)
        return (torch.from_numpy(augmented_signal), sr)
    
    # ----------------------------
    # Change the pitch of the audio file by factor shift_amount
    # without affecting the speed.
    # shift_amount determines the number of half-octaves to 
    # add/subtract depending on its sign
    # ----------------------------
    @staticmethod
    def pitch_shift(audio, shift_amount):
        signal, sr = audio
        augmented_signal = librosa.effects.pitch_shift(np.array(signal), sr=sr, n_steps=shift_amount)
        return (torch.from_numpy(augmented_signal), sr)
    
    # ----------------------------
    # Adjust volume of signal, increasing or decreasing overall loudness by factor 'gain_factor'
    # ----------------------------
    @staticmethod
    def adjust_volume(audio, gain_factor):
        signal, sr = audio
        transform = transforms.Vol(gain=gain_factor, gain_type='amplitude')
        return (transform(signal), sr)
    
    # ----------------------------
    # Apply random equalization profile, with maximal absolute change as 'max_eq' dB
    # ----------------------------
    def apply_equalization(audio, do_plot=False):
        signal, sr = audio
        
        num_samples = signal.shape[1]  # Length of the signal
        num_bands = np.random.randint(3, 10)  # Choose random number of bands
        
        # Convert to frequency domain using FFT
        spectrum = torch.fft.rfft(signal)

        # Generate a random equalization curve
        freqs = np.linspace(0, sr // 2, spectrum.shape[1])  # Frequency bins
        select_freqs = np.linspace(0, sr // 2, num_bands) # Select frequencies
        scale = np.linspace(0.05, 0.3, num_bands) # Scale for frequency dependent variance
        gains = np.abs(np.random.normal(loc=1, scale=scale, size=num_bands)) # Randomized gains at selected frequencies
        eq_curve = scipy.interpolate.make_interp_spline(select_freqs, gains, bc_type='clamped')(freqs)  # Random gain (log-normal dist)
        
        if do_plot:
            fig, ax = plt.subplots(figsize=(7,5))
            ax.plot(select_freqs, gains, 'bo', label='Randomized Anchor Points')
            ax.plot(freqs, eq_curve, label='Interprated Spline Curve')
            ax.set_ylim(eq_curve.min()-0.2, eq_curve.max()+0.2)
            ax.set_xlabel('Frequency (Hz)', fontsize=14) 
            ax.set_ylabel('Scaling Factor', fontsize=14) 
            ax.set_title('Randomized Equalization Curve', fontsize=16)
            ax.legend(fontsize=14) 
            ax.tick_params(axis='both', which='major', labelsize=12)
            plt.show()

        # Apply EQ curve (multiply in frequency domain)
        spectrum_eq = spectrum * np.float32(eq_curve)

        # Convert back to time domain
        signal_eq = torch.fft.irfft(spectrum_eq, n=num_samples)
        
        return (signal_eq, sr)
    
    # ----------------------------
    # (adaptive) High pass cutoff, from Jennie
    # ----------------------------
    @staticmethod
    def custom_high_pass_filter(audio, cutoff_freq='estimated_cutoff_freq', energy_threshold=0.15):
        signal, sr = audio

        # Appply filter channel by channel, and append to final tensor with all channels
        filtered_signal = []
        for i_channel in range(signal.size(0)):

            data = signal[i_channel,:].numpy()
            
            # 1 - estimate_high_pass_cutoff() from Jennies functions
            if cutoff_freq == 'estimated_cutoff_freq' and energy_threshold != 0:
                
                # Compute FFT
                N = len(data)
                freqs = np.fft.rfftfreq(N, d=1 / sr)  # Compute frequency bins
                fft_magnitude = np.abs(np.fft.rfft(data))  # Compute magnitude spectrum

                # Compute power spectrum
                power_spectrum = fft_magnitude ** 2

                # Normalize energy
                cumulative_energy = np.cumsum(power_spectrum) / np.sum(power_spectrum)

                # Find frequency where cumulative energy surpasses threshold
                cutoff_index = np.where(cumulative_energy > energy_threshold)[0][0]
                cutoff_freq = freqs[cutoff_index]

            # 2 - butter_highpass_filter() from Jennies functions
            if cutoff_freq > 0:
                order=4

                # Normalize the frequency with respect to Nyquist frequency
                nyquist_freq = sr/2
                normalized_cutoff = cutoff_freq / nyquist_freq

                # Design the Butterworth filter
                b, a = scipy.signal.butter(order, normalized_cutoff, btype='high', analog=False)

                # Apply the filter to the signal using filtfilt for zero-phase filtering
                filtered_data = scipy.signal.filtfilt(b, a, data)

            else:

                filtered_data = data

            # Add to full signal
            filtered_signal.append(filtered_data)

        filtered_signal = torch.tensor(np.array(filtered_signal)).float()

        return (filtered_signal, sr)

    # ----------------------------
    # Do pitch shift and time stretch with specified settings,
    # made for offline use. Also equalization if desired.
    # ----------------------------
    @staticmethod
    def offline_time_augment(audio, semitone_limits=[-2, 2], p_pitch_shift=0.5, 
                             stretch_rate_limits=[0.8, 1.25], p_time_stretch=0.5,
                             do_eq=True, do_at_least_one=True, do_print=False, return_settings=False):
        min_semitones, max_semitones = semitone_limits
        min_stretch, max_stretch = stretch_rate_limits

        # Equalization
        if do_eq:
            audio = AudioUtil.apply_equalization(audio)

        # Pitch shift. Print with time etc if do_print=True 
        do_pitch_shift = np.random.rand() < p_pitch_shift
        shift_amount = 0
        if do_print: print(f'Pitch shift: {do_pitch_shift}')
        if do_pitch_shift:
            shift_amount = np.random.uniform(min_semitones, max_semitones)
            start = time()
            audio = AudioUtil.pitch_shift(audio, shift_amount)
            end = time()
            if do_print: print(f'Shift amount = {shift_amount:.2f}, Time = {end-start:.4f}s')

        # Time stretch. Print with time etc if do_print=True    
        do_time_stretch = np.random.rand() < p_time_stretch
        do_time_stretch = do_time_stretch or do_at_least_one and not do_pitch_shift
        stretch_rate = 1
        if do_print: print(f'Time Stretch: {do_time_stretch}')
        if do_time_stretch:
            stretch_rate = np.random.uniform(min_stretch, max_stretch)
            start = time()
            audio = AudioUtil.time_stretch(audio, stretch_rate)
            end = time()
            if do_print: print(f'Stretch rate = {stretch_rate:.2f}, Time = {end-start:.4f}s')

        # Return extra parameters if requested
        if return_settings:
            return audio, shift_amount, stretch_rate
        else:
            return audio


    # ----------------------------
    # Generate a Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(audio, n_mels=64, n_fft=1024, hop_len=None):
        signal, sr = audio
        top_db = 80

        # sgram has shape [channel, n_mels, time], where channel is mono, stereo etc
        sgram = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(signal)

        # Convert to decibels
        sgram = transforms.AmplitudeToDB(top_db=top_db)(sgram)
        
        return (sgram)

    # ----------------------------
    # Augment the Spectrogram by masking out some sections of it in both the frequency
    # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
    # overfitting and to help the model generalise better. The masked sections are
    # replaced with the mean value.
    # ----------------------------
    @staticmethod
    def spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = sgram.shape
        mask_value = sgram.mean()
        aug_sgram = sgram

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_sgram = transforms.FrequencyMasking(freq_mask_param)(aug_sgram, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_sgram = transforms.TimeMasking(time_mask_param)(aug_sgram, mask_value)

        return aug_sgram
    


# Creates dataset for loading data into a spectrogram with augmentations, filter etc.
class SoundDS(Dataset):
    def __init__(self, df, data_path, duration, sr, hybrid_et=False, do_augment=False, filter_signal=False):
        self.df = df
        self.data_path = str(data_path)
        self.duration = duration*1000
        self.sr = sr
        self.channel = 1
        self.shift_pct = 1
        self.n_fft = 1024
        self.n_mels = 64
        self.hybrid_et = hybrid_et
        self.filter_signal = filter_signal
        self.do_augment = do_augment
            
    # ----------------------------
    # Number of items in dataset
    # ----------------------------
    def __len__(self):
        return len(self.df)    
    
    # ----------------------------
    # Get i'th item in dataset
    # ----------------------------
    def __getitem__(self, idx):

        # Absolute file path of the audio file - concatenate the audio directory with the relative path
        file_location = self.data_path + self.df.loc[idx, 'relative_path']
        if self.hybrid_et:

            # Select files
            audio_file_pre = file_location + self.df.loc[idx, 'file_name_pre']
            audio_file_post = file_location + self.df.loc[idx, 'file_name_post']

            # Read both files audio
            audio_pre = AudioUtil.open(audio_file_pre)
            audio_post = AudioUtil.open(audio_file_post)

            resampled_audio_pre = AudioUtil.resample(audio_pre, self.sr)
            rechanneled_audio_pre = AudioUtil.rechannel(resampled_audio_pre, self.channel)
            fixed_duration_audio_pre = AudioUtil.pad_trunc(rechanneled_audio_pre, self.duration)
            resampled_audio_post = AudioUtil.resample(audio_post, self.sr)
            rechanneled_audio_post = AudioUtil.rechannel(resampled_audio_post, self.channel)
            fixed_duration_audio_post = AudioUtil.pad_trunc(rechanneled_audio_post, self.duration)

            # Concatenate before and after et into the same tensor, as separate channel signals of the same length
            fixed_duration_audio = (torch.cat((fixed_duration_audio_pre[0], fixed_duration_audio_post[0]), dim=0), self.sr)

        else:

            # Select file
            audio_file = file_location + self.df.loc[idx, 'file_name']

            # Read file audio
            audio = AudioUtil.open(audio_file)

            audio = AudioUtil.open(audio_file)
            # Some sounds have a higher sample rate, or fewer channels compared to the
            # majority. So make all sounds have the same number of channels and same 
            # sample rate. Unless the sample rate is the same, the pad_trunc will still
            # result in arrays of different lengths, even though the sound duration is
            # the same.
            resampled_audio = AudioUtil.resample(audio, self.sr)
            rechanneled_audio = AudioUtil.rechannel(resampled_audio, self.channel)
            fixed_duration_audio = AudioUtil.pad_trunc(rechanneled_audio, self.duration)

        # Get the Class ID
        class_id = self.df.loc[idx, 'classID']

        # Filter signal if desired
        if self.filter_signal:
            filtered_audio = AudioUtil.custom_high_pass_filter(fixed_duration_audio)
        else:
            filtered_audio = fixed_duration_audio

        # Augment signal if desired, and turn into spectrogram
        if self.do_augment:

            time_shifted_audio = AudioUtil.time_shift(filtered_audio, self.shift_pct)

            if np.random.rand() > 0.0:
                noise_factor = np.abs(np.random.normal(loc=0, scale=1))
                noised_audio = AudioUtil.add_white_noise(time_shifted_audio, noise_factor=noise_factor)
            else: 
                noise_factor = 0
                noised_audio = time_shifted_audio
            
            sgram = AudioUtil.spectro_gram(noised_audio, n_mels=self.n_mels, n_fft=self.n_fft, hop_len=None)

            aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=np.random.randint(1,3), \
                                                                           n_time_masks=np.random.randint(1,3))

            return aug_sgram, class_id
        
        else:
            
            sgram = AudioUtil.spectro_gram(filtered_audio, n_mels=self.n_mels, n_fft=self.n_fft, hop_len=None)

            return sgram, class_id



def training(model, train_df, data_path, duration, sr, batch_size, val_dl, num_epochs, device, learning_rate, return_best_model, use_sliced_data, use_augmented_data, filter_signal=False, hybrid_et=False, separate_augments=True, do_plot=False):

    # Create training data sets (overwritten if using augmented data with separate_augments=True)
    train_ds = SoundDS(train_df, data_path, duration=duration, sr=sr, hybrid_et=hybrid_et, filter_signal=filter_signal, do_augment=True)

    # Create training data loader (overwritten if using augmented data with separate_augments=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)#, num_workers=4) 

    # Calculate the number of steps per epoch (overwritten if using augmented and sliced data with separate_augments=True)
    steps_per_epoch = int(len(train_dl))

    if use_augmented_data and separate_augments:

        n_augments = len(list(set(train_df['augment_number'])))

        if use_sliced_data:

            # Find the number of audio files in the augment 
            # folder with the least/most amount of files.
            len_of_shortest = np.inf
            len_of_longest = 0
            for i in range(n_augments):
                len_of_current = len(train_df[train_df['augment_number'].isin([i])])
                if len_of_current < len_of_shortest:
                    len_of_shortest = len_of_current
                if len_of_current > len_of_longest:
                    len_of_longest = len_of_current
            
            steps_per_epoch = int(len_of_longest/n_augments)
    
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate,
                                                    steps_per_epoch=steps_per_epoch,
                                                    epochs=num_epochs,
                                                    anneal_strategy='cos')

    # Prepare lists for plotting if requested
    if do_plot==True:
        loss_list = np.zeros(num_epochs)
        acc_list = np.zeros(num_epochs)
        val_acc_list = np.zeros(num_epochs)
        val_loss_list = np.zeros(num_epochs)

    highest_F1_score = -1
    highest_roc_auc = -1
    lowest_loss = np.inf
    bestModel = None

    # Repeat for each epoch
    for epoch in range(num_epochs):
        start = time()
        model.train()

        if use_augmented_data and separate_augments:

            # Randomly select one of the augment folders
            augment_df_mask = train_df['augment_number'].isin([int(np.random.randint(0,n_augments))])
            selected_train_df = train_df[augment_df_mask].reset_index(drop=True)
            rng = np.random.default_rng()

            # Select only some of the data, matching the fewest number of slices in all augment folders
            if use_sliced_data:
                chosen_index = rng.choice(len(selected_train_df), size=len_of_shortest, replace=False)
                selected_train_df = selected_train_df.iloc[chosen_index].reset_index(drop=True)

            # Create training data sets
            train_ds = SoundDS(selected_train_df, data_path, duration=duration, sr=sr, hybrid_et=hybrid_et, filter_signal=filter_signal, do_augment=True)

            # Create training data loaders
            train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)#, num_workers=4) 

        running_loss = 0
        
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):

            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)
            
            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(dim=(1, 2, 3), keepdim=True), inputs.std(dim=(1, 2, 3), keepdim=True)+1e-6
            inputs = (inputs - inputs_m) / inputs_s

            if torch.isnan(inputs).any():
                print("Warning: NaNs detected in inputs, applying nan_to_num!")
                inputs = torch.nan_to_num(inputs)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        # Keep stats for Loss and Accuracy
        num_batches = len(train_dl)
        
        avg_loss = running_loss / num_batches
        acc = correct_prediction/total_prediction

        if do_plot or return_best_model:
            loss_list[epoch] = avg_loss
            acc_list[epoch] = acc
            model.eval()
            val_acc_list[epoch], val_loss_list[epoch], F1_score, roc_auc = inference(model, val_dl, device, do_print=False)
            model.train()

            if F1_score > highest_F1_score and acc>=0.9:
                highest_F1_score = F1_score
                highest_roc_auc = 0

            if F1_score == highest_F1_score and acc>=0.9:

                if roc_auc > highest_roc_auc:
                    highest_roc_auc = roc_auc
                    lowest_loss = np.inf

                if roc_auc == highest_roc_auc:

                    if val_loss_list[epoch]+loss_list[epoch] < lowest_loss:
                        lowest_loss = val_loss_list[epoch]+loss_list[epoch]
                        bestModel = copy.deepcopy(model)

            elif epoch+1 == num_epochs and bestModel is None:
                bestModel = copy.deepcopy(model)
         
        end = time() 
        
        # Print stats at the end of each epoch  
        if do_plot: 
            print(f'Epoch: {epoch+1}, Loss: {avg_loss:.3f}, Accuracy: {acc:.3f}, Validation Loss: {val_loss_list[epoch]:.3f}, Validation Accuracy: {val_acc_list[epoch]:.3f}, F1 score: {F1_score:.3f}, roc-auc: {roc_auc:.3f}, Time: {end-start:.2f} s')
        else:       
            print(f'Epoch: {epoch+1}, Loss: {avg_loss:.3f}, Accuracy: {acc:.3f}, Time: {end-start:.2f} s')
    
    print('Finished Training')

    if return_best_model:
        model = copy.deepcopy(bestModel)

    if do_plot:

        # Plot accuracy over training
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,12), sharex=True)
        ax1.plot(acc_list, label='Training')
        ax1.plot(val_acc_list, label='Validation')

        ax1.legend()
        ax1.set_title('Accuracy throughout training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')

        ax2.plot(loss_list, label='Training')
        ax2.plot(val_loss_list, label='Validation')

        ax2.legend()
        ax2.set_title('Loss throughout training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Loss')
        ax2.set_ylim(bottom=-0.01)

        plt.show(block = False)
        plt.pause(5)
        plt.close('all')

        return model, fig, val_loss_list

    return None
    


def inference(model, val_dl, device, do_print=True, do_plot=False):
    
    criterion = nn.CrossEntropyLoss()
    correct_prediction = 0
    total_prediction = 0
    running_loss = 0
    
    prediction_list = []
    label_list = []
    y_pred_prob_list = []

    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:

            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(dim=(1, 2, 3), keepdim=True), inputs.std(dim=(1, 2, 3), keepdim=True)
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs,1)

            # Grab class=positive probability for ROC
            y_pred_prob = nn.functional.softmax(outputs,1)[:,1] # converting logits to probabilities
            
            # Append to lists
            prediction_list = np.append(prediction_list, np.array(prediction.cpu()))
            label_list = np.append(label_list, np.array(labels.cpu()))
            y_pred_prob_list = np.append(y_pred_prob_list, np.array(y_pred_prob.cpu()))

            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    # Confusion matrix
    if do_plot == True:

        confusion_matrix = metrics.confusion_matrix(label_list, prediction_list)

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = np.int32(list(set(label_list))))

        cm_display.plot()
        
        plt.show(block = False)
        plt.pause(5)
        plt.close('all')

    # Keep stats for loss, accuracy, f1 and roc
    num_batches = len(val_dl)
    avg_loss = running_loss / num_batches
    acc = correct_prediction/total_prediction
    f1_score = metrics.f1_score(label_list, prediction_list)
    roc_auc = ROC_analysis(label_list, y_pred_prob_list, do_plot=do_plot)

    # Print if desired
    if do_print == True:
        print(f'Loss: {avg_loss:.3f}, Accuracy: {acc:.3f}, F1 score: {f1_score:.3f}, roc-auc: {roc_auc:.3f}, Total items: {total_prediction}')

    return acc, avg_loss, f1_score, roc_auc



# Preprocess one file for evaluation
def preprocess_one_file(file_path, channel, split_options=[False, 0, 0], filter_signal=False):

    do_split, duration, slice_jump = split_options

    # Load audio file
    signal, sr = AudioUtil.open(file_path)

    if filter_signal:
        signal, sr = AudioUtil.custom_high_pass_filter((signal, sr))

    # Split audio into slices if requested
    audio_list = []
    if do_split:

        # Calculate start and end times for the slices
        start_times, end_times = calculate_slice_timings(0, len(signal[0])/sr, slice_jump, duration)

        # Slice the data
        for i_slice in range(len(start_times)):

            start_time = start_times[i_slice]
            end_time = end_times[i_slice]
            sliced_signal = signal[:,int(start_time*sr):int(end_time*sr)]
            sliced_audio = (sliced_signal, sr)

            audio_list.append(sliced_audio)

    else:

        audio_list.append((signal, sr))

    # Preprocess data and generate mel spectrogram for all audio signals
    spectrogram_list = []
    for audio in audio_list:
        
        # Resample, rechannel and fix duration
        resampled_audio = AudioUtil.resample(audio, 44100)
        rechanneled_audio = AudioUtil.rechannel(resampled_audio, channel)
        fixed_duration_audio = AudioUtil.pad_trunc(rechanneled_audio, duration*1000)

        # Generate spectrogram
        sgram = AudioUtil.spectro_gram(fixed_duration_audio, n_mels=64, n_fft=1024, hop_len=None)

        spectrogram_list.append(sgram)

    return spectrogram_list



# ---------------------------------------
#   Features model functions and classes
# ---------------------------------------



# Class for model trained on selected features
class FeaturesModel():
    def __init__(self, feature_matrix, labels, folds, cfg):

        self.X = feature_matrix
        self.encoder = LabelEncoder()
        self.y = labels
        self.folds = folds
        self.cfg = cfg
        self.model = cfg["model"]

        self.val_fold_scores_ = []

    def train_random(self, do_return_test=False, do_save_model=False, model_filename='sklearn_model', folder_path=Path.cwd()):

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)

        self.model = self.cfg['model']
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        fold_acc = metrics.accuracy_score(y_test, y_pred)
        self.val_fold_scores_.append(fold_acc)

        if do_save_model:
            self.save_model_as_onnx(filename=folder_path/'Models'/model_filename)

        if do_return_test: return self.val_fold_scores_, X_test, y_test
        else: return self.val_fold_scores_

    def train_kfold(self, do_save_each_model=False, model_filename='sklearn_model', folder_path=Path.cwd()):

        if do_save_each_model:
            Path(folder_path/'Models').mkdir(parents=True, exist_ok=True)

        logo = LeaveOneGroupOut()
        fold = 0
        for train_index, test_index in logo.split(self.X, self.y, self.folds):
            fold+=1

            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            self.model = self.cfg['model']
            self.model.fit(X_train, y_train)

            y_pred = self.model.predict(X_test)

            fold_acc = metrics.accuracy_score(y_test, y_pred)
            self.val_fold_scores_.append(fold_acc)

            if do_save_each_model:
                self.save_model_as_onnx(filename=folder_path/'Models'/(model_filename+f'_fold{fold}'))

        return self.val_fold_scores_
    
    def save_model_as_onnx(self, filename='demo2_model'):
        initial_type = [('float_input', FloatTensorType([None, np.shape(self.X)[1]]))]
        onx = convert_sklearn(self.model, initial_types=initial_type, options={'zipmap': False})
        with open(f'{filename}.onnx', 'wb') as f:
            f.write(onx.SerializeToString())



# Augment for feature extraction
def augment_for_feature_extraction(signal, sr, output_numpy=True):

    # Convert to torch.tensor, to be able to use previous augment definitions from torch model
    if signal.type() != torch.tensor(0.).type():
        audio = (torch.tensor(signal), sr)
    else:
        audio = (signal, sr)

    # Scale amplitude
    gain_factor = np.random.normal(loc=1, scale=0.1)
    audio = AudioUtil.adjust_volume(audio, gain_factor)

    # White noise
    #noise_factor = np.abs(np.random.normal(loc=0, scale=1))
    #audio = AudioUtil.add_white_noise(audio, noise_factor)

    # Equalization
    #audio = AudioUtil.apply_equalization(audio)

    # Time shift
    audio = AudioUtil.time_shift(audio, 1)

    signal, sr = audio

    if output_numpy: # Convert back to ndarray
        signal = (signal.numpy().T)[:,0]

    return signal


# Class for feature extraction
class AudioFeature:
    def __init__(self, src_path, fold, label, feature_names, pkl_path, augment=0, start_stop=[0,-1,-1]):
        self.filename = os.path.basename(src_path)
        self.src_path = src_path
        self.fold = fold
        self.label = label
        self.feature_names = feature_names
        self.pkl_path = pkl_path
        self.i_aug = int(augment)
        self.y, self.sr = AudioUtil.open(self.src_path)
        self.start, self.stop, self.i_slice = start_stop
        self.features = None

        # Limit the length if desired

        self.y = self.y[:,int(self.start*self.sr):int(self.stop*self.sr)] if self.i_slice != -1 else self.y

        # Interpret configuration settings
        _, _, _, _, _, _, filter_signal, normalize_signal = interpret_configuration(str(pkl_path))
 
        # Normalize signal if desired
        if normalize_signal == 'RMS':
            self.y, _ = AudioUtil.normalize_audio_by_RMS((self.y, self.sr)) 
        elif normalize_signal == 'Noise level':
            self.y, _ = AudioUtil.normalize_audio_by_noise_level((self.y, self.sr)) 

        # Apply filter if desired
        if filter_signal:
            self.y, _ = AudioUtil.custom_high_pass_filter((self.y, self.sr))

        # Do online augmentation if desired
        if int(self.i_aug)>0:
            self.y = augment_for_feature_extraction(self.y, self.sr, output_numpy=False)

        # Convert torch tensor to numpy array
        self.y = self.y.numpy()

    def _concat_features(self, feature):
        """
        Whenever a self._extract_xx() method is called in this class,
        this function concatenates to the self.features feature vector
        """
        self.features = np.hstack(
            [self.features, feature] if self.features is not None else feature
        )

    def _extract_mfcc(self, n_mfcc=25):
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=n_mfcc)

        mfcc_mean = mfcc.mean(axis=1).T
        mfcc_std = mfcc.std(axis=1).T
        mfcc_feature = np.hstack([mfcc_mean, mfcc_std])
        self._concat_features(mfcc_feature)

    def _extract_spectral_contrast(self, n_bands=3):
        spec_con = librosa.feature.spectral_contrast(
            y=self.y, sr=self.sr, n_bands=n_bands
        )

        spec_con_mean = spec_con.mean(axis=1).T
        spec_con_std = spec_con.std(axis=1).T
        spec_con_feature = np.hstack([spec_con_mean, spec_con_std])
        self._concat_features(spec_con_feature)

    def _extract_chroma_stft(self):
        stft = np.abs(librosa.stft(self.y))
        chroma_stft = librosa.feature.chroma_stft(S=stft, sr=self.sr)
        chroma_mean = chroma_stft.mean(axis=1).T
        chroma_std = chroma_stft.std(axis=1).T
        chroma_feature = np.hstack([chroma_mean, chroma_std])
        self._concat_features(chroma_feature)

    def _extract_opensmile(self):
        if self.i_aug == 0 and self.i_slice == -1:
            smile_features = smile.process_file(self.src_path) 
        else:
            smile_features = smile.process_signal(self.y, self.sr) # use if need to normalize
        smile_features = pd.DataFrame(smile_features)
        select_smile_features = smile_features[self.feature_names].values[0]
        select_smile_features = np.nan_to_num(select_smile_features)
        self._concat_features(select_smile_features)

    def extract_features(self, *feature_list, save_local=True):
        """
        Specify a list of features to extract, and a feature vector will be
        built for you for a given Audio sample.

        By default the extracted feature and class attributes will be saved in
        a local directory. This can be turned off with save_local=False.
        """
        extract_fn = dict(
            mfcc=self._extract_mfcc,
            spectral=self._extract_spectral_contrast,
            chroma=self._extract_chroma_stft,
            opensmile=self._extract_opensmile
        )

        for feature in feature_list:
            extract_fn[feature]()

        if save_local:
            self._save_local()

    def _save_local(self, clean_source=True):
        out_name = self.src_path.split('/')[-1]
        if self.i_slice != -1: out_name = f'{out_name}_slice{self.i_slice}'
        out_name = out_name.replace('.wav', '') + f'_online-aug{self.i_aug}'*(self.i_aug>0)

        #if self.fold == -1: filename = f'{self.pkl_path}/{out_name}.pkl'
        savepath = f'{self.pkl_path}/{out_name}.pkl'
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, 'wb') as f:
            pickle.dump(self, f)

        if clean_source:
            self.y = None



# Parse metadata for use of the files
def parse_metadata(path, variable_part_of_name, do_inference):

    # Read metadata file
    meta_df = pd.read_csv(path)
    
    # Check if the data has folds or not
    has_folds = False
    if 'folds' in meta_df.columns or 'fold' in meta_df.columns:
        has_folds = True

    # Interpret configuration settings
    _, use_augmented_data, use_exercise_test, before_and_after_et, use_2_classes, make_et_class, _, _ = interpret_configuration(variable_part_of_name)

    # If we are doing inference, we dont want to use augmented data anyway
    if do_inference:
        use_augmented_data = False

    # If the data has info on before/after exercise test
    if not 'missinget' in str(path):

        # Select audio files with 'exercise_test' = True or False depending on user input
        if not before_and_after_et:
            meta_df = meta_df[meta_df['exercise_test']==use_exercise_test].reset_index(drop=True)
        elif make_et_class:
            meta_df = meta_df[[type(et) == type(True) for et in meta_df['exercise_test']]]
    
    # Change classID to 0&1=0 and 2&3=1, or et
    if make_et_class and before_and_after_et:
        meta_df['classID'] = np.int64(meta_df['exercise_test'])
    elif use_2_classes: 
        meta_df['classID'] = np.int64(meta_df['classID']>1)

    # Rename 'slice_file_name' to 'file_name', for consistency
    if 'slice_file_name' in meta_df.columns:
        meta_df['file_name'] = meta_df['slice_file_name']
    
    if has_folds:

        # Take only the first fold in folds as the fold in which the data should lie, to prevent overlap
        if 'folds' in meta_df.columns:
            meta_df['fold'] = [int(folds[0]) for folds in meta_df['folds']]

        # Zip metadata
        if use_augmented_data:
            meta = zip(meta_df['file_name'], meta_df['fold'], meta_df['classID'], meta_df['augment_number'])
        else:
            meta = zip(meta_df['file_name'], meta_df['fold'], meta_df['classID'])
    
    else:
        
        # Zip metadata
        if use_augmented_data:
            meta = zip(meta_df['file_name'], meta_df['classID'], meta_df['augment_number'])
        else:
            meta = zip(meta_df['file_name'], meta_df['classID'])

    length = len(meta_df)

    return meta, length, has_folds



# Extract audio features 
def extract_audio_features(data_path, script_path, metadata_path, variable_part_of_name, suffix='', online_augments=0, slice_settings=[False,30,1], do_inference=False, single_file=False, save_features=True):
    # If using single_file = True, use metadata_path = filename of the single file

    do_slice, slice_length, slice_jump = slice_settings

    # Load selected feature names
    features_filename = f'feature_names{suffix}_{variable_part_of_name}.csv'
    features_path = script_path/'Feature data'/features_filename
    feature_names = pd.read_csv(features_path)['Feature']

    pkl_folder_name = f'training_pkl_data{suffix}_{variable_part_of_name}'
    if do_inference:
        pkl_folder_name = f'inference_pkl_data_{variable_part_of_name}'

    # Interpret configuration settings
    use_sliced_data, use_augmented_data, _, _, _, _, _, _ = interpret_configuration(variable_part_of_name)

    # If we are doing inference, we dont want to use augmented data anyway
    if do_inference:
        use_augmented_data = False

    # Parse metadata
    if single_file:
        filename = metadata_path
        metadata, n_data, has_folds = zip([filename], [0]), 1, False
    else:
        metadata, n_data, has_folds = parse_metadata(metadata_path, variable_part_of_name, do_inference)
 
    # Collect/Extract features
    audio_features = []
    
    for i, row in enumerate(metadata):

        print(f'Feature extraction progress: {100*i/n_data:.2f}%  ', end='\r')

        relative_save_folder = ''
        relative_src_folder = ''
        
        if use_augmented_data and use_sliced_data:
            filename, fold, label, augment_number = row
            relative_save_folder += f'augment{augment_number}/fold{fold}/'
            relative_src_folder += f'augment{augment_number}/fold{fold}/'

        elif use_augmented_data and has_folds:
            filename, fold, label, augment_number = row
            relative_save_folder = f'augment{augment_number}/fold{fold}/'
            relative_src_folder = f'augment{augment_number}/'

        elif use_augmented_data:
            filename, label, augment_number = row
            relative_save_folder = f'augment{augment_number}/'
            relative_src_folder = f'augment{augment_number}/'

        elif use_sliced_data and has_folds and do_slice:
            filename, fold, label = row
            relative_save_folder = f'fold{fold}/'

        elif use_sliced_data and has_folds:
            filename, fold, label = row
            relative_save_folder = f'fold{fold}/'
            relative_src_folder = f'fold{fold}/'

        elif use_sliced_data:
            filename, label = row
            fold = -1

        elif has_folds:
            filename, fold, label = row
            relative_save_folder = f'fold{fold}/'

        else: 
            filename, label = row
            fold = -1

        filename = filename.replace('.wav', '')
        transformed_path = f'{script_path}/Feature data/{pkl_folder_name}/{relative_save_folder}{filename}.pkl'
        src_path = f'{data_path}/{relative_src_folder}{filename}.wav'

        # Calculate start and end times for the slices
        if do_slice:
            signal, sr = librosa.load(src_path, mono=True)
            start_times, end_times = calculate_slice_timings(0, len(signal)/sr, slice_jump, slice_length)
        else:
            start_times = [0]
            end_times = [-1]
        
        for i_slice in range(len(start_times)):
        
            if do_slice: 
                transformed_path = f'{script_path}/Feature data/{pkl_folder_name}/{relative_save_folder}{filename}_slice{i_slice}.pkl'
                start_stop = [start_times[i_slice], end_times[i_slice], i_slice]
            else:
                start_stop = [start_times[0], end_times[0], -1]
            
            if os.path.isfile(transformed_path) and save_features:
                # if the file exists as a .pkl already, then load it
                with open(transformed_path, 'rb') as f:
                    audio = pickle.load(f)
                    audio_features.append(audio)
            else:
                # if the file doesn't exist, then extract its features from the source data and save the result
                audio = AudioFeature(src_path, fold, label, feature_names, pkl_path=script_path/'Feature data'/pkl_folder_name/relative_save_folder, augment=0, start_stop=start_stop)
                audio.extract_features('opensmile', save_local=save_features)
                audio_features.append(audio)

            # If online augmentaion is desired (online_augments > 0), do that too
            for i_aug in range(online_augments):
                i_aug += 1
                
                transformed_path_aug = transformed_path.replace('.pkl','') + f'_online-aug{i_aug}.pkl'
                
                if os.path.isfile(transformed_path_aug) and save_features:
                    # if the file exists as a .pkl already, then load it
                    with open(transformed_path_aug, 'rb') as f:
                        audio = pickle.load(f)
                        audio_features.append(audio)
                else:
                    # if the file doesn't exist, then extract its features from the source data and save the result
                    src_path = f'{data_path}/{relative_src_folder}{filename}.wav'
                    audio = AudioFeature(src_path, fold, label, feature_names, pkl_path=script_path/'Feature data'/pkl_folder_name/relative_save_folder, augment=i_aug, start_stop=start_stop)
                    audio.extract_features('opensmile', save_local=save_features)
                    audio_features.append(audio)

    print(f'Feature extraction progress: 100%    ')
    return audio_features



# ---------------------------------------
#   General functions and classes
# ---------------------------------------



# Read variable part of name to find configuration settings
def interpret_configuration(variable_part_of_name):

    # Check if the data is sliced or not
    use_sliced_data = False
    if not 'noslice' in variable_part_of_name:
        use_sliced_data = True

    # Check if the data is augmented or not
    use_augmented_data = False
    if not 'noaug' in variable_part_of_name:
        use_augmented_data = True
        
    # Check if the model is based on pre or post et data
    use_exercise_test = False
    before_and_after_et = False
    if 'bothet' in variable_part_of_name:
        before_and_after_et = True
    elif not 'noet' in variable_part_of_name:
        use_exercise_test = True

    # Check if the model uses only 2 classes or not
    use_2_classes = False
    make_et_class = False
    if '2class' in variable_part_of_name:
        use_2_classes = True
    elif 'etclass' in variable_part_of_name:
        make_et_class = True

    # Check if the model uses high pass filter or not
    filter_signal = False
    if '_filter' in variable_part_of_name:
        filter_signal = True

    # Check if the model uses any normalization
    normalize_signal = False
    if 'RMSNorm' in variable_part_of_name:
        normalize_signal = 'RMS'
    elif 'NoiseNorm' in variable_part_of_name:
        normalize_signal = 'Noise level'

    return use_sliced_data, use_augmented_data, use_exercise_test, before_and_after_et, use_2_classes, make_et_class, filter_signal, normalize_signal



# Calculate which parts of the signal should be put in what 
# folds to make sure there is no overlap between the folds.
def calculate_fold_timings(signal, folds, slice_jump, sr):

    # Determine length of signal and (approximately) fold.
    len_of_signal = np.shape(signal)[1]/sr # in seconds
    len_per_fold = len_of_signal/len(folds) # in seconds

    # Initiate lists onto which the start and end times will be appended
    start_times = []
    end_times = []

    for i, fold in enumerate(folds):

        # Determine the exact start and end times for the folds
        start_time = i*len_per_fold-(i*len_per_fold)%slice_jump
        end_time = (i+1)*len_per_fold-((i+1)*len_per_fold)%slice_jump

        # Append into lists
        start_times.append(start_time)
        end_times.append(end_time)

    # Make sure the last fold gets the rest of the signal, not cutting off the 
    # tail that exceeds slice_length, but not long enough for another slice_jump
    end_times[-1] = len_of_signal
    
    # Return the lists
    return start_times, end_times



# Calculate which parts of the signal should be put
# in which slice, based on fold start and end times.
def calculate_slice_timings(start_time, end_time, slice_jump, slice_length):

    # Initiate lists onto which the start and end times will be appended
    start_times = []
    end_times = []

    # Initiate start and end time iterables
    current_start_time = start_time
    current_end_time = start_time+slice_length

    # Loop until the end time exceeds the given end time
    while current_end_time <= end_time:

        # Append into lists
        start_times.append(current_start_time)
        end_times.append(current_end_time)

        # Itarate iterables
        current_start_time += slice_jump
        current_end_time += slice_jump

    # Make sure the last slice gets the rest of the signal, not cutting off the 
    # tail that exceeds slice_length, but not long enough for another slice_jump
    end_times[-1] = end_time
    
    return start_times, end_times



# ROC Analysis
def ROC_analysis(y_test, y_prediction_probabilities, do_plot=True):

    # Compute the false positive rate (FPR)  
    # and true positive rate (TPR) for different classification thresholds 
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_prediction_probabilities, pos_label=1)

    # Compute the ROC AUC score 
    roc_auc = metrics.roc_auc_score(y_test, y_prediction_probabilities)

    # Plot the ROC curve 
    if do_plot:
        fig, ax = plt.subplots(figsize=(7,5))
        ax.plot(fpr, tpr, label=f'ROC Curve (Area = {roc_auc:.2f})') 
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier (Area = 0.5)') 
        ax.set_xlabel('False Positive Rate', fontsize=14) 
        ax.set_ylabel('True Positive Rate', fontsize=14) 
        ax.set_title('ROC Curve', fontsize=16) 
        ax.legend(loc='lower right', fontsize=14) 
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.show(block = False)
        plt.pause(5)
        plt.close('all')

    return roc_auc