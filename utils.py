import os
import glob
import inspect


def find_files(data_type, folder_path):
    
    ''' 
    Finds files of the specified data tpye (search for the first letter)
    
    Parameters:
        data_type (str): Data type to search for in the file name
        folder_path (str): Path to the folder to search in
            
    Return:
        List of full paths of files containing the specified data type
    '''
    
    pattern = os.path.join(folder_path, data_type + "*")
    files = glob.glob(pattern)
    return files


def update_flag(main_folder, data_type):
    
    '''
    Update the processing status flag file based on the caller function and data type.
    
    Parameters:
        main_folder (str): The path to the main folder where the flag file is located.
        data_type (str): The type of data being processed. It can be either 'T' for training data or 'E' for evaluation data.

    '''

    # define the flag file
    flag_file = os.path.join(main_folder, "processing_status.txt")
    
    # read the current content of the flag file
    with open(flag_file, "r") as f:
        content = f.read()
    
    # get the caller function name
    caller = inspect.stack()[1].function
    
    # update the content based on the caller function name and data type
    if caller == 'load_eeg_data':
        if data_type == 'T':  # Training data type
            content = content.replace("training_imported=False", "training_imported=True")
        elif data_type == 'E':  # Evaluation data type
            content = content.replace("evaluation_imported=False", "evaluation_imported=True")
    elif caller == 'clean_eeg_data':
        if data_type == 'T':  # Training data type
            content = content.replace("training_preprocessed=False", "training_preprocessed=True")
        elif data_type == 'E':  # Evaluation data type
            content = content.replace("evaluation_preprocessed=False", "evaluation_preprocessed=True")

    with open(flag_file, "w") as f:
        f.write(content)
    
    
# Function to check if data is imported
def is_data_imported(main_folder, data_type):
    
    # define the flag file
    flag_file = os.path.join(main_folder, "processing_status.txt")
    prefix = "training" if data_type == "T" else "evaluation"
    
    with open(flag_file, "r") as f:
        for line in f:
            if line.strip().startswith(f"{prefix}_imported"):
                return line.split("=")[-1].strip() == "True"
    return False

# Function to check if data is preprocessed
def is_data_preprocessed(main_folder, data_type):
    
    # define the flag file
    flag_file = os.path.join(main_folder, "processing_status.txt")
    prefix = "training" if data_type == "T" else "evaluation"
    
    with open(flag_file, "r") as f:
        for line in f:
            if line.strip().startswith(f"{prefix}_preprocessed"):
                return line.split("=")[-1].strip() == "True"
    return False