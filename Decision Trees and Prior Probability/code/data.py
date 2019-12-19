import numpy as np 
import os
import csv

def load_data(data_path):
    """
    loads data from data path csv into two numpy arrays representing features and targets
    
    assumptions:
    - data_path leads to a csv comma-delimited file with each row corresponding to a 
    different example. 
    - Each row contains binary features for each example 
    - The last column indicates the label for the attributes
    
    Args:
        data_path (str): path to csv file containing data

    Output:
        features (np.array): numpy array of size NxK containing the K features
        targets (np.array): numpy array of size Nx1 containing the 1 feature.
        attribute_names (list): list of strings containing names of each attribute (i.e. headers of csv)
    """

    with open(data_path, newline='') as csvfile:
        csv_data = list(csv.reader(csvfile, delimiter=','))

        attribute_names = csv_data[0][:-1]
        
        features = []
        targets = []
        for row in csv_data[1:]:
            features.append(row[:-1])
            targets.append(row[-1])
        
        feature_floats = np.asarray([list(map(float,row)) for row in features])
        target_floats = np.asarray(list(map(int,targets)))
        
        
    return feature_floats, target_floats, attribute_names
             

def train_test_split(features, targets, fraction):
    """
    Split features and targets into training and testing subsections based on fraction 
    specified for training data.
    
    Args:
        features (np.array): numpy array containing features for each example
        targets (np.array): numpy array containing labels corresponding to each example.
        fraction (float): fraction of examples to be drawn for training

    Returns
        train_features, train_targets, test_features, test_targets
    """
    
    if (fraction > 1.0):
        raise ValueError('N cannot be bigger than number of examples')
    elif (fraction < 0):
        raise ValueError('N cannot be negative')
    elif (fraction == 1):
        train_features = features 
        test_features = features
        train_targets = targets
        test_targets = targets
    else:    
        N_train = int(features.shape[0] * fraction)
        N_test = features.shape[0] - N_train
        
        # check that the fraction split the dtaa such that it is still used in its entirety
        assert N_test + N_train == features.shape[0]
        
        list_of_ones_for_train_set = [1] * N_train
        list_of_zeros_for_test_set = [0] * N_test
        
        import random
        
        unshuffled_indices_for_train = list_of_ones_for_train_set + list_of_zeros_for_test_set
        
        shuffled_indices_for_train =  np.array(random.sample(unshuffled_indices_for_train, len(unshuffled_indices_for_train)))
        
        train_features = features[shuffled_indices_for_train==1]
        train_targets = targets[shuffled_indices_for_train==1]
        test_features = features[shuffled_indices_for_train==0]
        test_targets = targets[shuffled_indices_for_train==0]
    
    return train_features, train_targets, test_features, test_targets
    
    
    
    
    
    