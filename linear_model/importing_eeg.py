import numpy as np
from scipy.io import loadmat
import mne
import os
from utils import update_flag


def load_eeg_data(main_folder, source_path, raw_folder, data_type):
    
    """
    Load EEG data from the source .mat file and save it as mne(.fif) file
    
    Parameters:
        source_path (str): Path to the source .mat file
        data_type (str): Type of the data 'T': training 'E': evaluation 
        
    """  
    
    # load the data
    mat_file    = loadmat(source_path)
    source_data = mat_file['data'][0]
    
    # each data consists of different sessions
    # for trainig there are 3 sessions, for evaluation there are 2 sessions
    
    # Inside each section (example): 
    # X:       467479x6 -----> EEG data: time x channel
    # trial:   160x1    -----> trial latencies
    # y:       160x1    -----> class information
    # fs:      250      -----> sampling frequency

    # classess:
    # 1: left hand
    # 2: right hand
    
    #     mne: epoch x channel x time
    # raw mne: channel x time
    
    # loop over sessions
    for s, session_data in enumerate(source_data, start=1):
        
        # extract EEG data in each session
        EEG_session = session_data['X'][0][0] # time x channel
        EEG_session = np.transpose(EEG_session) # channel x time 
        
        # extract trial latency info
        trials = session_data['trial'][0][0]
        
        # extract classification info for left-right hand imagination
        classes = session_data['y'][0][0];
        
        
        # create info objects for mne file
        sampling_freq = 250 # sampling frequency
        chan_names = ['EEG1','EEG2','EEG3','EOG1','EOG2','EOG3'] # channel names to identify EOG channels
        channel_types = ['eeg', 'eeg', 'eeg','eog', 'eog', 'eog']
        info = mne.create_info(ch_names=chan_names, sfreq=sampling_freq, ch_types=channel_types)
        
        
        # create mne raw EEG structure 
        EEG = mne.io.RawArray(EEG_session, info)
        
        # create event matrix: latency x 0 x class
        events = np.column_stack((trials, np.zeros_like(trials), classes))
        
        # add stim channel
        info_stim = mne.create_info(['STIM'], EEG.info['sfreq'], ['stim'])
        stim_data = np.zeros((1, len(EEG.times)))
        stim_raw  = mne.io.RawArray(stim_data, info_stim)
        EEG.add_channels([stim_raw], force_update_info=True)
        
        # add the events to the stim channel
        EEG.add_events(events)
    
        # construct the file name with data type and session number
        raw_filename = f'{data_type}_session{s}_EEG.fif'
        
        raw_path = os.path.join(raw_folder,raw_filename) # path
        
        # save session raw EEG data
        EEG.save(raw_path)

    # Update the flag to indicate that data is imported
    update_flag(main_folder, data_type)
    