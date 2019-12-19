import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

def adjusted_mutual_info(predicted_labels, predicted_targets):
    """
    Wrapping function for adjusted mutual info score from sk-learn to judge 
    the quality of a clustering output compared to available ground truths
    """
    return adjusted_mutual_info_score(predicted_labels, predicted_targets)