import numpy as np

def euclidean(a,b):
    return np.linalg.norm(a-b)
    
class KMeans():
    def __init__(self, n_clusters):
        """
        Implements traditional KMeans algorithm with hard assignments using two steps

        1. Update assignments
        2. Update the means
        
        Args:
            n_clusters (int): Number of clusters to cluster the given data into
        """
        self.n_clusters = n_clusters
        self.means = None


    def fit(self, features):
        """
        Fit KMeans to the given multidimensional data with as many clusters as specified

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features)
        """
        
        assignments = np.zeros(features.shape[0], dtype=int)
        
        self.random_means(features)
        
        while not np.array_equal(assignments, self.update_assignments(features)):
            assignments = self.update_assignments(features)
            self.update_means(features,assignments)
            
        
    
    def random_means(self,features):
        random_indices = np.random.randint(features.shape[0], size=self.n_clusters)
        self.means = np.zeros((self.n_clusters, features.shape[1]))
        
        for cluster_id in range(self.n_clusters):
            self.means[cluster_id,:] = features[random_indices[cluster_id],:]
        
        
    def update_assignments(self, features):
        
        assignments = np.zeros(features.shape[0], dtype=int)
        
        for sample_id in range(features.shape[0]):
            point = features[sample_id,:]
            #closest_mean_index = float('inf')
            #closest_mean_dist = float('inf')
            cluster_distances = np.zeros(self.n_clusters)
            
            for cluster_id in range(self.n_clusters):
                cluster_distances[cluster_id] = euclidean(point, self.means[cluster_id,:])
            assignments[sample_id] = np.argmin(cluster_distances)
        return assignments
        
      
    def update_means(self,features,assignments):
        
        sum_of_cluster_features = np.zeros((self.n_clusters, features.shape[1]))
        tally_of_cluster_counts = np.zeros((self.n_clusters,1), dtype=int)
        for sample_id in range(features.shape[0]):
            sum_of_cluster_features[assignments[sample_id]] = np.add(sum_of_cluster_features[assignments[sample_id]], features[sample_id])
            tally_of_cluster_counts[assignments[sample_id]] += 1
            
        tally_of_cluster_counts[np.where(tally_of_cluster_counts==0)] = 1
        self.means = np.divide(sum_of_cluster_features, tally_of_cluster_counts)
        
        

    def predict(self, features):
        """
        predicts feature cluster membership by comparing points to the means saved

        Args:
            features (np.ndarray): array containing inputs of size (n_samples, n_features)
        Returns:
            predictions (np.ndarray): predicted cluster membership for each feature
        """
        return self.update_assignments(features)
        