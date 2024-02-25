import numpy as np
import mne

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import find_files


def extract_features(data_type, cleaned_folder):
     
    
    # create empty feature matrix
    feature_matrix = np.array([])  # store features for each trials
    # Column: Features of trials: total 216 columns (72 x 3)
    # Row: Sample data: Feature vector of each trial 
    
    class_vector = np.array([]) # store info about motor imagery classes
    
    # find all training or evaluation cleaned files
    cleanedEEG_files = find_files(data_type, cleaned_folder)
    
    # loop over cleaned files
    for cleanedfile_path in cleanedEEG_files:
        
        # load the data
        cleanedEEG = mne.io.read_raw_fif(cleanedfile_path, preload=True)
    
        # 1. Generate a filter bank
        #---------------------------
    
        # EEG Motor Imagery Frequency Band: 8-30 Hz
        # Mu Frequency: 8:13 Hz Beta Frequency: 14-30 Hz
    
        # Number of overlapping bands:72 Overlap Width:1 Hz
        # 21: width 2 Hz
        # 19: width 4 Hz
        # 17: width 6 Hz
        # 15: width 8 Hz
    
        frequency_bands = np.arange(8, 30)  # Frequency bands
        band_widths = [2, 4, 6, 8]  # Overlap widths
        number_overlaps = [21, 19, 17, 15]  # Number of overlapping bands per width
        
        session_features = np.array([]) # store feature in each session
        
        # loop over band widths
        for Wi, width in enumerate(band_widths):
            
            # loop over frequency bands
            for Fi in range(number_overlaps[Wi]):
                
                low_cutoff = frequency_bands[Fi]  # overlap width: 1 Hz
                high_cutoff = low_cutoff + width
                #iir_params = dict(ftype = 'cheby2')
                
                # Apply band-pass filter
                cleanedEEG.filter(l_freq=low_cutoff, h_freq=high_cutoff)
                
                # 2. Epoching
                #-------------------
                # 0-3 sec: Baseline
                # 3-4 sec: Cue 
                # 4-7 sec: Motor Imagery
    
                # Cue phase continues during the motor imagery in evaluation sessions but
                # this does not change the interested epochs.
    
                # get the events from the data
                events = mne.find_events(cleanedEEG)
                
                # select the epoch period: baseline, cue, motor imagery
                tmin = 2.8
                tmax = 7
                
                # define event ids
                event_id = {'left hand': 1, 'right hand': 2}
                
                # define baseline period
                baseline = (None, 3)
                
                # only pick eeg channels and exclude eog channels
                picks = mne.pick_types(cleanedEEG.info, eeg=True, eog=False)
                       
                # segment the data into 4 seconds epochs 
                epochedEEG = mne.Epochs(cleanedEEG, events=events, event_id=event_id, tmin=tmin,
                    tmax=tmax, baseline=baseline, picks=picks)
                
                
                # 3. Calculate ERD/ERS
                #----------------------
                
                # extract sample data from the epochs
                epochedEEG_data = epochedEEG.get_data() # trial x channel x sample
                
                # Square the signal to compute the power
                sqr_epoched = epochedEEG_data**2

                # average it over time
                avg_epoched = np.mean(sqr_epoched, axis=2) 

                # add each frequency filter as a feature/column
                if len(session_features) == 0:
                    session_features = avg_epoched
                else:
                    session_features = np.hstack((session_features, avg_epoched))
                
        # add each session trils to feature matrix           
        if len(feature_matrix) == 0:
            feature_matrix = session_features
        else:
            feature_matrix = np.vstack((feature_matrix, session_features))
        
        # find class info in the session
        classes = events[:,2]
        
        # add this to class vector
        if len(class_vector) == 0:
            class_vector = classes
        else:
            class_vector = np.hstack((class_vector, classes))
    
    # change the class names as 0 and 1 instead of 1 and 2
    mapping = {1: 0, 2: 1}

    # Apply the mapping to your labels
    class_vector = np.array([mapping[label] for label in class_vector])
    
    # the filter_bank function returns these two arrayss
    return feature_matrix, class_vector


def select_features(feature_matrix):
    
    # normalize the data
    scaler = StandardScaler()
    feature_matrix = scaler.fit_transform(feature_matrix)
    
    # perform PCA
    pca = PCA(n_components = 0.95)
    selected_features = pca.fit_transform(feature_matrix)
    
    return selected_features, pca