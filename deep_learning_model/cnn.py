import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import keras_tuner as kt

from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def cnn_training(cnn_feature_matrix, cnn_class_vector):
    
    input_shape = cnn_feature_matrix.shape[1:]
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(cnn_feature_matrix, cnn_class_vector, test_size=0.2, random_state=42)
    
    # Define the hyperparameter search space
    def model_builder(hp):
        model = Sequential()
        # Convolutional layer 1
        model.add(Conv1D(filters=hp.Choice('filters', [64, 128, 256]), kernel_size=hp.Choice('kernel_size', [3, 5]), activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))

        # Convolutional layer 2
        model.add(Conv1D(filters=hp.Choice('filters', [64, 128, 256]), kernel_size=hp.Choice('kernel_size', [3, 5]), activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

        # Convolutional layer 3
        model.add(Conv1D(filters=hp.Choice('filters', [64, 128, 256]), kernel_size=hp.Choice('kernel_size', [3, 5]), activation='relu'))
        model.add(MaxPooling1D(pool_size=2))

        # Flatten layer to transition from convolutional layers to dense layers
        model.add(Flatten())

        # Dense layers
        model.add(Dense(units=hp.Choice('dense_units', [256, 512]), activation='relu'))
        model.add(Dropout(rate=hp.Choice('dropout_rate', [0.3, 0.5])))

        # Output layer
        model.add(Dense(2, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model
    
    # Initialize the Keras Tuner
    tuner = kt.Hyperband(model_builder,
                         objective='val_accuracy',
                         max_epochs=10,
                         hyperband_iterations=3,
                         directory='my_dir',
                         project_name='cnn_tuning')
    
    # Perform hyperparameter tuning
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    # Build the model with the best hyperparameters
    best_model = tuner.hypermodel.build(best_hps)
    
    # Train the best model
    best_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))
    
    # Evaluate the best model on the validation set
    _, accuracy = best_model.evaluate(X_val, y_val)
    print("Validation Accuracy of CNN Model:", accuracy)
    
    # Return the best model
    return best_model

def cnn_testing(main_folder, best_model, cnn_feature_matrix, cnn_class_vector):
    
    # Evaluate the model on the testing set
    _, test_accuracy = best_model.evaluate(cnn_feature_matrix, cnn_class_vector)
    print("Testing Accuracy of CNN Model:", test_accuracy)
    
    
    ## ROC Curve
    #----------------------
    
    # Predict probabilities for positive class
    y_pred_proba = best_model.predict(cnn_feature_matrix)

    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(cnn_class_vector, y_pred_proba[:, 1])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(12, 12))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for CNN Model')
    plt.legend(loc="lower right")
    plt.show()
    
    # save the figure
    roc_plot_name = 'Roc_Curve_CNN.png'
    roc_file_path = os.path.join(main_folder, roc_plot_name)
    plt.savefig(roc_file_path)
    
    ## Confusion Matrix
    #-----------------------
   
    # Get the predicted classes
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Generate confusion matrix
    cm = confusion_matrix(cnn_class_vector, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(16, 12))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=['Left Hand', 'Right Hand'], yticklabels=['Left Hand', 'Right Hand'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix - CNN')
    plt.show()
    
    # Save plot to file
    conf_plot_name = 'Confusion_Matrix_CNN.png'
    conf_file_path = os.path.join(main_folder, conf_plot_name)
    plt.savefig(conf_file_path)
    
    return test_accuracy