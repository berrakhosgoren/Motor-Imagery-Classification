# Motor Imagery Classification

The project processes EEG signals recorded from a subject imagining left and right-hand movements. It implements two different machine learning approaches (linear model and deep learning model) to classify motor imagery patterns.

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
