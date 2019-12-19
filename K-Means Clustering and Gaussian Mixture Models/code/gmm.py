import numpy as np
from code import KMeans
from scipy.stats import multivariate_normal

class GMM():
    def __init__(self, n_clusters, covariance_type):
        """
        Gaussian Mixture Model that is updated using expectation maximization
        with two steps

        1. Update assignments to each Gaussian
        2. Update Gaussian means and variances
        
        Args:
            n_clusters (int): Number of Gaussians to cluster the given data into
            covariance_type (str): covariance type for the Gaussians in the mixture model.
            either 'spherical' or 'diagonal'
        """
        self.n_clusters = n_clusters
        allowed_covariance_types = ['spherical', 'diagonal']
        if covariance_type not in allowed_covariance_types:
            raise ValueError(f'covariance_type must be in {allowed_covariance_types}')
        self.covariance_type = covariance_type

        self.means = None
        self.covariances = None
        self.mixing_weights = None
        self.max_iterations = 200

    def fit(self, features):
        """
        Fit GMM to the given multidimensional data with as many gaussians as self.n_clusters

        Args:
            features (np.ndarray): input array of size (n_samples, n_features).
        """
        # 1. Use your KMeans implementation to initialize the means of the GMM.
        kmeans = KMeans(self.n_clusters)
        kmeans.fit(features)
        self.means = kmeans.means

        # 2. Initialize the covariance matrix and the mixing weights
        self.covariances = self._init_covariance(features.shape[-1])

        # 3. Initialize the mixing weights
        self.mixing_weights = np.random.rand(self.n_clusters)
        self.mixing_weights /= np.sum(self.mixing_weights)

        # 4. Compute log_likelihood under initial random covariance and KMeans means.
        prev_log_likelihood = -float('inf')
        log_likelihood = self._overall_log_likelihood(features)

        # 5. While the log_likelihood is increasing significantly, or max_iterations has
        # not been reached, continue EM until convergence.
        n_iter = 0
        while log_likelihood - prev_log_likelihood > 1e-4 and n_iter < self.max_iterations:
            prev_log_likelihood = log_likelihood

            assignments = self._e_step(features)
            self.means, self.covariances, self.mixing_weights = (
                self._m_step(features, assignments)
            )

            log_likelihood = self._overall_log_likelihood(features)
            n_iter += 1

    def predict(self, features):
        """
        predicts feature label based on cluster to which it belongs and corresponding label

        Args:
            features (np.ndarray): input array of size (n_samples, n_features)
        Returns:
            predictions (np.ndarray): predicted assigment to each cluster for each sample
        """
        posteriors = self._e_step(features)
        return np.argmax(posteriors, axis=-1)

    def _e_step(self, features):
        """
        expectation step in EM, three components:
            1. Calculate log_likelihood of point under each Gaussian
            2. Calculate posterior probability for each point under each Gaussian
            3. Return posterior probability (assignments)
        
        Arguments:
            features {np.ndarray} -- Features to find assignments for

        Returns:
            np.ndarray -- Posterior probabilities to each Gaussian 
        """
        posterior_probabilities = np.zeros((features.shape[0], self.n_clusters))
        
        for cluster_id in range(self.n_clusters):
            posterior_probabilities[:,cluster_id] = self._posterior(features, cluster_id)
        
        return posterior_probabilities
                
        
    def _m_step(self, features, assignments):
        """
        Maximization step in Expectation-Maximization. Implement update equations 
        for means, covariances and mixing weights for gaussians
            
        Arguments:
            features {np.ndarray} -- Features to update means and covariances given assignments
            assignments {np.ndarray} -- Soft assignments of each point to one of the clusters from e_step

        Returns:
            means -- Updated means
            covariances -- Updated covariances
            mixing_weights -- Updated mixing weights
        """
        
        cluster_weights = []
        cluster_means = []
        cluster_covariances = []
        
        for cluster_id in range(self.n_clusters):
            gamma_j = 0
            means_numerator = 0
            sigma_numerator = 0
            for feature,feature_id in zip(features,range(len(features))):
                gamma_j += assignments[feature_id,cluster_id]
                means_numerator += assignments[feature_id,cluster_id]*feature
                sigma_numerator += assignments[feature_id,cluster_id]*(feature-self.means[cluster_id])**2
            
            w_j = gamma_j/len(features)
            mu_j = means_numerator/gamma_j
            sigma_j = sigma_numerator/gamma_j
            
            cluster_weights.append(w_j)
            cluster_means.append(mu_j)
            cluster_covariances.append(sigma_j)
            
        return np.asarray(cluster_means), np.asarray(cluster_covariances), np.asarray(cluster_weights)

        

    def _init_covariance(self, n_features):
        """
        Initialize the covariance matrix given the covariance_type (spherical or
        diagonal). If spherical, each feature is treated the same (has equal covariance).
        If diagonal, each feature is treated independently (n_features covariances).

        Arguments:
            n_features {int} -- Number of features in the data for clustering

        Returns:
            [np.ndarray] -- Initial random covariances
        """
        if self.covariance_type == 'spherical':
            return np.random.rand(self.n_clusters)
        elif self.covariance_type == 'diagonal':
            return np.random.rand(self.n_clusters, n_features)

    def _log_likelihood(self, features, k_idx):
        """
        Compute the likelihood of the features given the index of the Gaussian
        in the mixture model
        
        Arguments:
            features {np.ndarray} -- Features to compute multivariate_normal distribution
            k_idx {int} -- Which Gaussian to use 
            
        Returns:
            np.ndarray -- log likelihoods of each feature given a Gaussian
        """
        return (np.log(self.mixing_weights[k_idx]) + multivariate_normal.logpdf(features, mean=self.means[k_idx], cov=self.covariances[k_idx]))

    def _overall_log_likelihood(self, features):
        denom = [
            self._log_likelihood(features, j) for j in range(self.n_clusters)
        ]
        return np.sum(denom)

    def _posterior(self, features, k):
        """
        Computes the posteriors given the log likelihoods for the GMM

        Arguments:
            features {np.ndarray} -- Numpy array containing data as (n_samples, n_features)
            k {int} -- Index of which Gaussian to compute posteriors for

        Returns:
            np.ndarray -- Posterior probabilities for the selected Gaussian k, of size (n_samples,)
        """
        num = self._log_likelihood(features, k)
        denom  = np.array([
            self._log_likelihood(features, j)
            for j in range(self.n_clusters)
        ])

        max_value = denom.max(axis=0, keepdims=True)
        denom_sum = max_value + np.log(np.sum(np.exp(denom - max_value), axis=0))
        posteriors = np.exp(num - denom_sum)
        return posteriors
