import numpy as np 
import random

def generate_regression_data(degree, N, amount_of_noise=1.0):
    """
    Used to verify linear and polynomial regressors. Generates arrays of length N 
    with a polynomial relationship to the degree specified. The amount of noise is 
    the randomize noise added to the perfect polynomial relationship generated

    Args:
        degree (int): degree of polynomial relating x and y
        N (int): number of points
        amount_of_noise (float): amount of random noise to added to polynoial relationship
        
    Returns:
        x (np.ndarray): independent variable between -1 and +1
        y (np.ndarray): dependent variable
    """
    
    unscaled_x_data = np.asarray(range(N))
    scaled_x_data = (unscaled_x_data*2/(N-1))-np.asarray([1]*N)
    
    coefficients = [random.randrange(-10,10) for coefficient in range(degree)]
    
    y_data = []
    for x_datum in scaled_x_data:
        components_for_y_datum = []
        for power in range(degree):
            components_for_y_datum.append(coefficients[power]*x_datum**power)
        components_for_y_datum = np.asarray(components_for_y_datum)
        y_datum = np.sum(components_for_y_datum)
        y_data.append(y_datum)
    y_data = np.asarray(y_data)
    y_with_noise = y_data + np.random.normal(loc=0.0, scale=np.std(y_data)*amount_of_noise, size=y_data.shape)
    
    return np.asarray(scaled_x_data), y_with_noise
