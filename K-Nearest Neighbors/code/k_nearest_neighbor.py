import numpy as np 
from collections import Counter

from .distances import euclidean_distances, manhattan_distances, cosine_distances


class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        Implements K nearest neighbor algorithm for recommendation systems

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for predictions
            distance_measure {str} -- Distance metric to use, can be euclidean, manhattan or cosine 
            aggregator {str} -- how to aggregate neighbor values to form prediction
        """
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator
        


    def fit(self, features, targets):
        """
        imports features and target arrays and puts them in as model features
        """

        self.train_features = features
        self.train_targets = targets
        

    def predict(self, features, ignore_first=False):
        """
        takes each point in featur array, finds nearest neighbors within training feature and
        aggregates their targets from training targets to report predictions as the aggregate

        Arguments:
            features {np.ndarray} -- Features of each data point of shape (n_samples,n_features).
            ignore_first {bool} -- Ignore the point itself if true, include it if false. determines
            if point itself is used as a neighbor and so is useful for collaborative filtering

        Returns:
            labels {np.ndarray} -- Labels for each data point of shape (n_samples,n_features)
        """
        
        
        # measures distance as specified 
        if self.distance_measure == "euclidean":
            distances_reference = euclidean_distances(features,self.train_features)
        elif self.distance_measure == "manhattan":
            distances_reference = manhattan_distances(features,self.train_features)
        elif self.distance_measure == "cosine":
            distances_reference = cosine_distances(features,self.train_features)
        
        
        labels = []
        
        for sample_idx in range(features.shape[0]):
            
            # find an ordered list of the closest n samples to it, including or excluding itself as specified
            if ignore_first == True:
                idx_nearest_neighbors = np.argpartition(distances_reference[sample_idx,:], self.n_neighbors-1)[1:self.n_neighbors]
            else:
                idx_nearest_neighbors = np.argpartition(distances_reference[sample_idx,:], self.n_neighbors-1)[:self.n_neighbors]
            
            targets_nearest_neighbors = self.train_targets[idx_nearest_neighbors]
            
            # the label assembled is taken from the aggregated valued of the neighbors and appended to the list of labels
            # the label assembled is taken from the aggregated valued of the neighbors and appended to the list of labels
            if self.aggregator == "mean":
                label = np.mean(targets_nearest_neighbors, axis=0)
            elif self.aggregator == "mode":
                label = self.helper_mode(targets_nearest_neighbors)
            elif self.aggregator == "median":
                label = np.median(targets_nearest_neighbors, axis=0)            
            labels.append(label)
            
        return np.asarray(labels)
        

    def helper_mode(self, to_be_imputed):
        from collections import Counter
        mode_row = np.zeros(to_be_imputed.shape[1])
        for ii, col in enumerate(to_be_imputed.T):
            mode_row[ii] = Counter(col).most_common(1)[0][0]
        return mode_row    

