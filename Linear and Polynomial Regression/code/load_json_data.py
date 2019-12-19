import json
import numpy as np 
import os

def load_json_data(json_path):
    """
    loads json data from file path

    Args:
        json_path (str): path to json file
    Returns:
        features (np.ndarray): numpy array of x values
        targets (np.ndarray): numpy array of y values, either -1 or 1.
    """

    with open(json_path, 'rb') as f:
        data = json.load(f)
    features = np.array(data[0]).astype(float)
    targets = 2 * (np.array(data[1]).astype(float) - 1) - 1

    return features, targets