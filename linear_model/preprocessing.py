import numpy as np
import mne
import os
from mne.preprocessing import EOGRegression
from utils import find_files, update_flag


def clean_eeg_data(main_folder, raw_folder, cleaned_folder, data_type):
    
    '''
    Preprocess the raw EEG data with mne scripts
    
    '''
    
    # find all training or evaluation raw files
    rawEEG_files = find_files(data_type, raw_folder)
    
    # loop over raw session files
    for rawfile_path in rawEEG_files:
        
        # load the data
        rawEEG = mne.io.read_raw_fif(rawfile_path, preload=True)
        
        # 1. Reduce power line noise
        #----------------------------
        # notch filter at 50 Hz
        notch_freq = 50  # frequency
        bandwidth = 1    # bandwidth
        rawEEG.notch_filter(freqs=notch_freq, notch_widths=bandwidth)

        # 2. Remove baseline drift 
        #--------------------------
        # high pass filter at 0.5 Hz
        rawEEG.filter(l_freq=0.5, h_freq=None)

        # 3. Bandpass filter between 2-60 Hz
        #------------------------------------
        iir_params = dict(order=5, ftype='butter') # 5th order butterworth
        rawEEG.filter(l_freq=2, h_freq=60, method='iir', iir_params=iir_params)

        # 4. Detecting outliers
        #-----------------------
        threshold = 6  # threshold for outlier detection

        # number of channels
        num_channels = rawEEG.info['nchan'] 

        # loop over EEG channels
        for ch in range(num_channels - 1):  # exclude the last stim channel
        
            # Calculate mean and standard deviation for each channel
            mean_channel = np.mean(rawEEG.get_data()[ch,:])
            std_channel  = np.std(rawEEG.get_data()[ch, :])

            # Clip outliers beyond the threshold: find the outliers and set it to threshold value
            rawEEG._data[ch, rawEEG._data[ch, :] > mean_channel + threshold * std_channel] = mean_channel + threshold * std_channel
            rawEEG._data[ch, rawEEG._data[ch, :] < mean_channel - threshold * std_channel] = mean_channel - threshold * std_channel

            # Normalize the data: subtract the mean value from the data and divide it by standard deviation
            rawEEG._data[ch, :] = (rawEEG._data[ch, :] - mean_channel) / std_channel
          
        
        # 5. Detect and remove EOG artifacts
        #------------------------------------------
        # MNE: Reduce EOG artifacts through regression
        
        # define the EOG channels
        eog_channels = ['EOG1', 'EOG2', 'EOG3']
        
        # Set EEG channels reference to average reference
        rawEEG.set_eeg_reference(ref_channels='average', projection=False, ch_type='eeg')

        
        # Fit the regression model
        weights    = EOGRegression(picks_artifact=eog_channels).fit(rawEEG)
        cleanedEEG = weights.apply(rawEEG, copy=True)
        
        
        ## Save the cleaned EEG data 
        
        # find the raw file name
        raw_filename = os.path.basename(rawfile_path)
        file_name, extension = os.path.splitext(raw_filename)
        
        # define the new name
        cleaned_filename = file_name + '_cleaned_' + '.fif'
        
        cleaned_path = os.path.join(cleaned_folder,cleaned_filename) # path
        
        # save cleaned EEG data in each loop
        cleanedEEG.save(cleaned_path)
        
    # Update the flag to indicate that data is imported
    update_flag(main_folder, data_type)