import numpy as np
from .k_nearest_neighbor import KNearestNeighbor


def collaborative_filtering(input_array, n_neighbors,
                            distance_measure='euclidean', aggregator='mode'):
    """
    Takes all unspecified values as zero values and then imputes them using the
    aggregator in the KNN model. This is used to form a collaborative filtering model

    Arguments:
        input_array {np.ndarray} -- An input array of shape (n_samples, n_features).
            Any zeros will get imputed.
        n_neighbors {int} -- Number of neighbors to use for prediction
        distance_measure {str} -- Which distance measure to use. Can be euclidean, 
        manhattan or cosine
        aggregator {str} -- How to aggregate labels of neighbors, either mode, mean or median

    Returns:
        imputed_array {np.ndarray} -- An array of shape (n_samples, n_features) with imputed
            values for any zeros in the original input_array
    """
    # copy of input array to be modified
    imputed = input_array
    
    # we run the nearest neighbors function where the targets are the same as the input arrays themselves
    # then the labels we get are just the aggregated/imputed versions of those inputs
    nearest_neighbors_class = KNearestNeighbor(n_neighbors, distance_measure, aggregator)
    nearest_neighbors_class.fit(input_array,input_array)
    aggregated = nearest_neighbors_class.predict(input_array)
    
    # we replace the zero values in the input array with their analogues from the aggregated labels
    for input_row_idx in range(input_array.shape[0]):
        if 0 in input_array[input_row_idx,:]:
            imputing_idx = np.where(input_array[input_row_idx,:]==0)
            imputed[input_row_idx,:][imputing_idx] = aggregated[input_row_idx,:][imputing_idx]
    
    return imputed    
    
    
