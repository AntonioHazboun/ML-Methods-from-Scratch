import numpy as np

def mean_squared_error(estimates, targets):
    """
    Measures average of the square of the errors 
    MSE = (1 / n) * \sum_{i=1}^{n} (Y_i - Yhat_i)^2
    """
    
    MSE = ((targets-estimates)**2).mean()
        
    return MSE