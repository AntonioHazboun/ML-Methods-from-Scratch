import numpy as np
try:
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

class PolynomialRegression():
    def __init__(self, degree):
        """
        Polynomial regression based on linear regression in which the X data is raised to the degree
        specified and every degree below that to zero.
        Assumes 1 dimension of input data and output data
        
        Args:
            degree (int): Degree of polynomial to which X data is raised
        """
        self.degree = degree
    
    def fit(self, features, targets):
        """
        Fit polynomial regressor to data. Raises X data to self.degree and changes weights 
        to fit to targets
        
        Args:
            features (np.ndarray): 1D array of inputs
            targets (np.ndarray): 1D array of outputs
        """
        
        one_D_features = np.array(features)
        
        z_data = []
        X = []
        for power in range(self.degree + 1):
            z_data.append(one_D_features)
            X.append(list(map(lambda x:pow(x,power),one_D_features)))
        
        self.training_X = np.asarray(features)
        self.training_Y = np.asarray(targets)
        
        X = np.asarray(X).transpose()
        
        w = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(targets).flatten()
        
        self.weights = w

    def predict(self, features):
        """
        Uses trained model to return predictions from input features

        Args:
            features (np.ndarray): 1D array of inputs.
        Returns:
            predictions (np.ndarray): Output of saved model on features
        """
        X_data = []
        self.x_data = X_data
        for power in range(self.degree + 1):
            X_data.append(list(map(lambda x:pow(x,power),features)))
        
        X_data = np.asarray(X_data).transpose()
        
        Y = X_data.dot(self.weights)
        
        return Y
    
    def visualize(self, features, targets):
        """
        Produces scatter plot of features and best fit line from polynomial function 
        produced after fittings and saves the figure

        Args:
            features (np.ndarray): 1D array of inputs
            targets (np.ndarray): 1D array of targets
        """
        
        Z = features
        Y_predictions = self.predict(features)
        Y_targets = targets
        
        plt.scatter(Z, Y_targets)
        plt.plot(Z, Y_predictions)
        
        plt.savefig()
        
        
        