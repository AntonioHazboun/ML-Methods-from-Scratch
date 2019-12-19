import numpy as np 

def euclidean_distances(X, Y):
    """
    Calculates euclidean distance between points in two matrices and retuns third matrix
    of distances
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples by K features
        Y {np.ndarray} -- Second matrix, containing N examples by K features

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """

    D = np.zeros((X.shape[0],Y.shape[0]))
    
    for X_idx in range(X.shape[0]):
        for Y_idx in range(Y.shape[0]):  
            
            D[X_idx,Y_idx] = np.sqrt(np.sum((X[X_idx,:]-Y[Y_idx,:])**2))
    
    return D        
    


def manhattan_distances(X, Y):
    """
    same as euclidean distance function but with manhattan distance
    """
    D = np.zeros((X.shape[0],Y.shape[0]))
    
    for X_idx in range(X.shape[0]):
        for Y_idx in range(Y.shape[0]):  
            
            D[X_idx,Y_idx] = np.sum(np.abs(X[X_idx,:] - Y[Y_idx,:]))
    
    return D


def cosine_distances(X, Y):
    """
        same as euclidean distance function but with cosine distance
    """
    D = np.zeros((X.shape[0],Y.shape[0]))
    
    for X_idx in range(X.shape[0]):
        for Y_idx in range(Y.shape[0]):  
            
            D[X_idx,Y_idx] = 1 - (np.dot(X[X_idx,:],Y[Y_idx,:]) / (np.sqrt(np.dot(X[X_idx,:], X[X_idx,:]))* np.sqrt(np.dot(Y[Y_idx,:], Y[Y_idx,:]))))            
    return D
