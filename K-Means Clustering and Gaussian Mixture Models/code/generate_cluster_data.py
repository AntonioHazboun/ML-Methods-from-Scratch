import numpy as np
from sklearn.datasets import make_blobs

def generate_cluster_data(n_samples=100,n_features=2,n_centers=2,cluster_stds=1.0):
    """
    Generate numpy arrays that are clusterable into specified number of clusters. It is
    a wrapping function for make_blobs from sk_learn
    
    Args:
        n_samples (int): Number of samples to generate
        n_features (int): Number of features for each sample
        n_centers (int): Number of clusters to generate
        cluster_stds (float or sequence of floats): standard deviation for each cluster.
    
    Returns:
        X (np.ndarray of shape (n_samples, n_features): A numpy array containing the
            generated data. Each row represents a point in n_features-dimensional space.

        y (np.ndarray of shape (n_samples,): A numpy array containing the cluster labels
            for the generated data. Each element tells you which cluster each data point
            came from. 
    """
    return make_blobs(n_samples=n_samples, n_features=n_features, centers=n_centers, cluster_std=cluster_stds)
