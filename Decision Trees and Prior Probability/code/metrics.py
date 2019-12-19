import numpy as np

def confusion_matrix(actual, predictions):
    """
    returns 2*2 confusion matrix assuing binary data values as per

    [
        [true_negatives, false_positives],
        [false_negatives, true_positives]
    ]


    Args:
        actual (np.array): actual labels of length N
        predictions (np.array): predicted labels of length N

    Output:
        confusion_matrix (np.array): 2x2 confusion matrix between predicted and actual labels
    """
    
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length")
    
    true_negatives = 0
    false_negatives = 0
    false_positives = 0
    true_positives = 0
    
    for actual_value, predicted_value in zip(actual,predictions):
        if int(actual_value) == 0 and int(predicted_value) == 0:
            true_negatives += 1
        elif int(actual_value) == 1 and int(predicted_value) == 0:
            false_negatives += 1
        elif int(actual_value) == 0 and int(predicted_value) == 1:
            false_positives += 1
        else:
            true_positives += 1
            
    confusion_matrix = [[true_negatives, false_positives],[false_negatives, true_positives]]
    
    return confusion_matrix

def accuracy(actual, predictions):
    """
    computes the accuracy by referencing the confusion matrix function from above, must be defined together!
    """

    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")
    
    CM = confusion_matrix(actual, predictions)
    
    acc = (CM[1][1]+CM[0][0])/(CM[1][1]+CM[0][0]+CM[1][0]+CM[0][1])
    
    return acc

def precision_and_recall(actual, predictions):
    """
    returns precision and recall using confusion matrix above as per wiki definition
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length")

    CM = confusion_matrix(actual, predictions)
    #[[TN, FP],[FN, TP]]
    
    if CM[1][1] != 0:        
        precision = CM[1][1]/(CM[1][1]+CM[0][1])
        recall = CM[1][1]/(CM[1][1]+CM[1][0])
    else:
        precision = 0
        recall = 0
    
    """
    precision = (conf_matrix[1, 1])/(conf_matrix[1, 1] + conf_matrix[0, 1])
    recall = (conf_matrix[1, 1])/(conf_matrix[1, 1] + conf_matrix[1, 0])
    """
    
    return precision, recall

def f1_measure(actual, predictions):
    """
    returns f1 measure from wiki definition
    """
    if predictions.shape[0] != actual.shape[0]:
        raise ValueError("predictions and actual must be the same length!")

    precision, recall = precision_and_recall(actual, predictions)
    
    if (precision + recall) != 0:
        F1_score = (2*precision*recall)/(precision + recall)
    else:
        F1_score = 0
        
    return F1_score

