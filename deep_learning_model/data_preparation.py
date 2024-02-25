import os
import mne
import numpy as np

from utils import is_data_imported, find_files
from linear_model.importing_eeg import load_eeg_data

def epoch_rawEEG(main_folder, source_folder, raw_folder, data_type):
    
    # Importing
    #-----------
    
    # check whether the importing was done or not before
    if not is_data_imported(main_folder, data_type):
        
        if data_type == 'T':
            file_name = r'B04T.mat'
        elif data_type == 'E':
            file_name = r'B04E.mat'
        
        source_path = os.path.join(source_folder, file_name)
        
        # import the training data
        load_eeg_data(source_path, raw_folder, data_type)

    else:
        print("Data is already imported. Skipping data importing process.")
        
    # Epoching
    #----------
    
    epoched_data_list = [] # list to store all epoched data
    cnn_class_vector  = np.array([]) # array to store class info
    
    # find all training or evaluation raw files
    rawEEG_files = find_files(data_type, raw_folder)
    
    # loop over raw session files
    for rawfile_path in rawEEG_files:
        
        # load the data
        rawEEG = mne.io.read_raw_fif(rawfile_path, preload=True)
        
        # get the events from the data
        events = mne.find_events(rawEEG)
        
        # select the epoch period: motor imagery
        tmin = 4
        tmax = 7
        
        # define event ids
        event_id = {'left hand': 1, 'right hand': 2}
        
        # only pick eeg channels and exclude eog channels
        picks = mne.pick_types(rawEEG.info, eeg=True, eog=False)
               
        # segment the data into 3 seconds epochs 
        epochedEEG = mne.Epochs(rawEEG, events=events, event_id=event_id, tmin=tmin,
            tmax=tmax, baseline=None, picks=picks)
        
        # get the data
        epoched_data = epochedEEG.get_data()
        
        # append the data in to the list
        epoched_data_list.append(epoched_data)
        
        # find class info in the session
        classes = events[:,2]
        
        # add this to class vector
        if len(cnn_class_vector) == 0:
            cnn_class_vector = classes
        else:
            cnn_class_vector = np.hstack((cnn_class_vector, classes))
    
    # change the class names as 0 and 1 instead of 1 and 2
    mapping = {1: 0, 2: 1}

    # Apply the mapping to your labels
    cnn_class_vector = np.array([mapping[label] for label in cnn_class_vector])
        
    # concatenate the arrays along the first dimension
    cnn_feature_matrix = np.concatenate(epoched_data_list, axis=0)
    
    # transpose the data to epoch x sample x electrode from epoch x electrode x sample
    cnn_feature_matrix = np.transpose(cnn_feature_matrix, (0, 2, 1))
    
    return cnn_feature_matrix, cnn_class_vector
    