import json
import numpy as np
import os

def load_json_data(json_path):
    """
    Loads data from JSON files kept in data/. Implemented this for you, you are
    welcome.

    Args:
        json_path (str): path to json file containing the data (e.g. 'data/blobs.json')
    Returns:
        features (np.ndarray): numpy array containing the x values
        targets (np.ndarray): numpy array containing the y values in the range -1, 1.
    """

    with open(json_path, 'rb') as f:
        data = json.load(f)
    features = np.array(data[0]).astype(float)
    targets = 2 * (np.array(data[1]).astype(float) - 1) - 1

    return features, targets
