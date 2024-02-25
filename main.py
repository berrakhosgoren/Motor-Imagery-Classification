import os
import sys

## SETUP
#------------------------------------------------------------------------------

# Add the parent directory of the current script to the Python path
parent_dir = os.path.dirname(os.path.abspath(r'C:\Users\Berrak\Documents\GitHub\neuroengineering\Task-1\Python\main.py'))
sys.path.append(parent_dir)

# set folder paths
main_folder    = r'C:\Users\Berrak\Documents\GitHub\neuroengineering\Task-1\Python' 
source_folder  = r'C:\Users\Berrak\Documents\GitHub\neuroengineering\Task-1\EEG-data\1_SourceData-EEG'
raw_folder     = r'C:\Users\Berrak\Documents\GitHub\neuroengineering\Task-1\EEG-data\2_RawData-EEG'
cleaned_folder = r'C:\Users\Berrak\Documents\GitHub\neuroengineering\Task-1\EEG-data\3_CleanedData-EEG'


# Import necessary modules from linear_model directory
from linear_model.importing_eeg import load_eeg_data
from linear_model.preprocessing import clean_eeg_data
from linear_model.feature_engineering import extract_features, select_features
from linear_model.LDA import train_lda, classify_lda

# Import necessary modules from deep_learning_model directory
from deep_learning_model.data_preparation import epoch_rawEEG
from deep_learning_model.cnn import cnn_training, cnn_testing


# Import utility functions
from utils import is_data_imported, is_data_preprocessed


## FLAG FILE: to check preprocessing and importing status of datasets
#------------
# create the flag file to keep track of processing status 
flag_file = os.path.join(main_folder, "processing_status.txt")

# define the default status
default_status = (
"training_imported=False\n"
"training_preprocessed=False\n"
"evaluation_imported=False\n"
"evaluation_preprocessed=False\n"
)

# Check if the flag file already exists if not create it
if not os.path.exists(flag_file):
    with open(flag_file, "w") as f:
        f.write(default_status)



def EEG_linear_model():
    
    
    # TRAINING
    #-------------------------------------------------------------------------
    
    data_type = 'T' # training
    
    # Data Processing
    feature_matrix_tra, class_vector_tra = EEG_linear_data_processing(main_folder, source_folder, raw_folder, cleaned_folder, data_type)
    
    # Feature Selection
    selected_features_tra, pca = select_features(feature_matrix_tra)
    
    # Train the data with Linear Discriminant Analysis (LDA)
    lda = train_lda(selected_features_tra, class_vector_tra)
    
    
    # TESTING
    #--------------------------------------------------------------------------
    
    data_type = 'E' # evaluation
    
    # Data Processing
    feature_matrix_eva, class_vector_eva = EEG_linear_data_processing(main_folder, source_folder, raw_folder, cleaned_folder, data_type)
    
    # Feature Selection
    selected_features_eva = pca.transform(feature_matrix_eva)
    
    # Test the model with the evaluation dataset
    accuracy_lda = classify_lda(main_folder, lda, selected_features_eva,class_vector_eva)
    
    return accuracy_lda

def EEG_linear_data_processing(main_folder, source_folder, raw_folder, cleaned_folder, data_type):
    
    # Step 01: Importing Files 
    #--------------------------
    
    # check whether the importing was done or not before
    if not is_data_imported(main_folder, data_type):
        
        if data_type == 'T':
            file_name = r'B04T.mat'
        elif data_type == 'E':
            file_name = r'B04E.mat'
        
        source_path = os.path.join(source_folder, file_name)
        
        # import the training data
        load_eeg_data(main_folder, source_path, raw_folder, data_type)

    else:
        print("Data is already imported. Skipping data importing process.")
    
    
    # Step 02: Preprocessing the Data
    #---------------------------------
    
    # check whether the preprocessing was done or not before
    if not is_data_preprocessed(main_folder, data_type):
        
        # preprocessed the raw data
        clean_eeg_data(main_folder, raw_folder, cleaned_folder, data_type)

    else:
        print("Data is already preprocessed. Skipping the preprocessing step.")   
    

    # Step 03: Feature Extraction
    #----------------------------
    
    feature_matrix, class_vector = extract_features(data_type, cleaned_folder)
    
    
    return feature_matrix, class_vector


def EEG_deep_learning_model():

    # TRAINING
    #-------------------------------------------------------------------------
    
    data_type = 'T' # training
    
    # epoch the training data into 3 second segments
    cnn_feature_matrix_t, cnn_class_vector_t = epoch_rawEEG(main_folder, source_folder, raw_folder, data_type)
    
    # train the data with the cnn model
    best_model = cnn_training(cnn_feature_matrix_t, cnn_class_vector_t)
    
    # TESTING
    #--------------------------------------------------------------------------
    data_type = 'E' # evaluation
    
    # epoch the evaluation data into 3 second segments
    cnn_feature_matrix_e, cnn_class_vector_e = epoch_rawEEG(main_folder, source_folder, raw_folder, data_type)
    
    # evaluate the model
    test_accuracy = cnn_testing(main_folder, best_model, cnn_feature_matrix_e, cnn_class_vector_e)
    
    return test_accuracy

if __name__ == "__main__":
    
    accuracy_lda  = EEG_linear_model()  
    test_accuracy = EEG_deep_learning_model()
    
    print('Accuracy of the linear model is:', accuracy_lda)
    print('Accuracy of the deep learning model is:', test_accuracy)