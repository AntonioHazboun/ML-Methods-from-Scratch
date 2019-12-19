import numpy as np
import math
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

def transform_data(features):
    """
    tTransforms data into polar coordinates for circular data that is not linearly separable
 
    Args:
        features (np.ndarray): input features
    Returns:
        transformed_features (np.ndarray): features in polar coordinates
    """
    data = np.asarray(features)
    X_untransformed = data[:,0]
    Y_untransformed = data[:,1]
    
    X_mean = np.mean(X_untransformed)
    Y_mean = np.mean(Y_untransformed)
    
    X_mean_centered = X_untransformed.copy()
    Y_mean_centered = Y_untransformed.copy()
    
    X_mean_centered[X_mean_centered != np.nan] -= X_mean
    Y_mean_centered[Y_mean_centered != np.nan] -= Y_mean
    
    polar_features = []
    for datapoint in range(len(features)):
        r = math.sqrt(X_mean_centered[datapoint]**2 + Y_mean_centered[datapoint]**2)
        theta = np.arctan2(Y_mean_centered[datapoint],X_mean_centered[datapoint])
        
        polar_datapoint = [r,theta]
        
        polar_features.append(polar_datapoint)
    
    return polar_features
        
    
class Perceptron():
    def __init__(self, max_iterations=200):
        """
        Implements linear discriminator as perceptron. label_for_example is either -1 or 1.

        Args:
            max_iterations (int): the perceptron learning algorithm stops after 
            this many iterations if it has not converged
        """
        self.max_iterations = max_iterations
        self.weights = None
        
    
    def predict_individual_point(self,point):
        predicted_value = self.weights.T.dot(point)
        if predicted_value > 0:
            return 1
        else:
            return -1
    
    def fit(self, features, targets):
        """
        Fit a single layer perceptron to features to classify the targets. terminates
         after convergence or max iteration number is reached
            
        Args:
            features (np.ndarray): 2D array of inputs
            targets (np.ndarray): 1D array of binary targets
        """
        feature_array = np.asarray(features)
        number_of_inputs = feature_array.shape[1]

        self.weights = np.zeros(number_of_inputs+1)

        x = [np.insert(features[ii],0,1) for ii in range(feature_array.shape[0])]

        for iteration_count in range(self.max_iterations):
            for point_idx in range(len(x)):
                
                target = targets[point_idx]
                
                if target == self.predict_individual_point(x[point_idx]):
                    continue
                else:
                    self.weights = self.weights + (target*x[point_idx])
                
                
        
    def predict(self, features):
        """
        Use trained model to predict classes of features 
        
        Args:
            features (np.ndarray): 2D array of inputs
        Returns:
            predictions (np.ndarray): predicted feature classifications
        """
        predictions = []
        feature_array = np.asarray(features)
        
        x = [np.insert(features[ii],0,1) for ii in range(feature_array.shape[0])]
        
        for point_idx in range(len(x)):
            predicted_value = self.predict_individual_point(x[point_idx])
            
            predictions.append(predicted_value)
        
        return np.asarray(predictions)

    def visualize(self, features, targets):
        """
        Produces scatter plot of features and best fit line from polynomial function 
        produced after fittings and saves the figure

        Args:
            features (np.ndarray): 1D array of inputs
            targets (np.ndarray): 1D array of targets
        """
        
        self.fit(features, targets)
    
        plt.figure(figsize=(6,4))
        plt.scatter(features[:, 0], features[:, 1], c=targets)

        plt.plot(features[:,0], self.weights.T.dot(point))
        
        plt.savefig()
