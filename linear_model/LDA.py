import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

def train_lda(selected_features_tra,class_vector_tra):
    
    
    # initialize the LDA model
    lda = LinearDiscriminantAnalysis()
    
    # fit the model on the training data
    lda.fit(selected_features_tra, class_vector_tra)

    return lda

def classify_lda(main_folder, lda, selected_features_eva,class_vector_eva):

    # conduct classification on the evaluation data
    predicted_labels = lda.predict(selected_features_eva)
    
    # evaluate the model
    accuracy_lda = metrics.accuracy_score(class_vector_eva, predicted_labels)
    
    # print the accuracy
    print('Accuracy of the linear model is:', accuracy_lda)
    
    ## ROC Curve
    #----------------------
    
    # calculate posterior probabilities
    posteriors = lda.predict_proba(selected_features_eva)
    
    # compute ROC curve and ROC area (AUC) for each class
    fpr, tpr, thresholds = metrics.roc_curve(class_vector_eva, posteriors[:, 1])
    roc_auc = metrics.auc(fpr, tpr)
    
    # plot roc curve
    plt.figure(figsize=(12, 12))
    plt.plot(fpr, tpr, lw=2, label='ROC curve (area = {0: 0.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], lw=2, c = 'k', linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve for LDA Model', fontweight='bold', fontsize=18)
    plt.legend(loc="lower right");
    
    # save the figure
    roc_plot_name = 'Roc_Curve_LDA.png'
    roc_file_path = os.path.join(main_folder, roc_plot_name)
    plt.savefig(roc_file_path)


    ## Confusion Matrix
    #-----------------------
    
    # Create confusion matrix
    C = confusion_matrix(class_vector_eva, predicted_labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(16, 12))
    sns.heatmap(C, annot=True, cmap='Blues', fmt='g', xticklabels=['Left Hand', 'Right Hand'], yticklabels=['Left Hand', 'Right Hand'])
    plt.title('Confusion Matrix - LDA')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=0)
    plt.yticks(rotation=90)
    plt.tight_layout()
    
    # Save plot to file
    conf_plot_name = 'Confusion_Matrix_LDA.png'
    conf_file_path = os.path.join(main_folder, conf_plot_name)
    plt.savefig(conf_file_path)
    
    return accuracy_lda