import numpy as np

class PriorProbability():
    def __init__(self):
        """
        Uses prior probability approach to classify points. It looks at the classes 
        for each data point and returns the most common class
        """
        self.most_common_class = None

    def fit(self, features, targets):
        """
        Fits prior probability decision model to the data.

        Args:
            features (np.array): NxF array with N examples of F features
            targets (np.array): Array with N targets corresponding to features
        """

        class_counts = np.bincount(targets)
        
        self.most_common_class = np.argmax(class_counts)

    def predict(self, data):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): NxF array with N examples of F features
        """
        
        return np.asarray([self.most_common_class]*len(data))