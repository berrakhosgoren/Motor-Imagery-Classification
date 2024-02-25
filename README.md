# Motor Imagery Classification

The project processes EEG signals recorded from a subject imagining left and right-hand movements. It implements two different machine learning approaches (linear model and deep learning model) to classify motor imagery patterns.

# Code Structure

The project's code is organized as follows:

- `main.py`: This script serves as the main entry point for the project. It controls the execution flow and orchestrates the linear and deep learning model.
- `utils.py`: This script contains helper functions used across both linear and deep learning models.
  
- **linear_model**: This directory contains the implementation of the linear model.
  - `importing.py`: Script for importing data to mne files. 
  - `preprocessing.py`: Script for preprocessing the data before feeding it into the linear model.
  - `feature_engineering.py`: Script for feature extraction and selection.
  - `LDA.py`: Script containing the implementation of the Linear Discriminant Analysis (LDA) classifier.
    
- **deep_learning_model**: This directory contains the implementation of the deep learning model.
  - `data_preparation.py`: Script for preparing the data specifically for the deep learning model.
  - `CNN.py`: Script containing the implementation of the convolutional neural network (CNN) and hyperparameter tuning.


## Linear Model

### Data Processing 
The data was preprocessed and cleaned to remove noise and artifacts. It was then segmented into 4-second epochs for analysis.

### Feature Extraction 
Filter bank method was used to extract features.

### Feature Selection 
Principal component analysis (PCA) was applied to select features.

### Classifier 
Linear Discriminant Analysis (LDA)

## Deep Learning Model

### Data Processing
Raw data was segmented into 3-second epochs.

### Feature Extraction
No feature extraction was performed for the deep learning model, as raw data was used.

### Classifier
Convolutional Neural Network (CNN)
