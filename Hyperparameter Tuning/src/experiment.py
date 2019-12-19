
from .grid_search import GridSearchCV
from .mnist import load_mnist
from .circle import load_circle
from .random_search import RandomSearchCV


def run(estimator, search_type, tuned_parameters, dataset, n_iter=5):
    '''
    loads either the MNISt dataset or the circular classification dataset
    upon which the parameter tuning module will be tested. 
    generates list of parameter combinations either with random or grid search as
    defined in other coding files and reports on the results of the parameter tuning 
    comparisons. returns a list of all configurations created by param tuner with the 
    resulting accuracy.
    
    args:
        - estimator: Estimator chosen, will be SVM by default
        - search_type: method by which combinations of parameters are formed
        - tuned_parameters: Contains a dictionary with parameters to tune and
          the tuning values that should be tested by the tuner.
        - dataset: Dataset upon which parameter comparison will be done (MNIST or circular)
        - n_iter: The number of iterations for the Random Search algorithm
    '''

    NUMBER_OF_MNIST_SAMPLES = 500

    # Load the right dataset depending on the dataset param
    if dataset == 'mnist':
        inputs, targets = load_mnist()
        inputs = inputs[:NUMBER_OF_MNIST_SAMPLES]
        targets = targets[:NUMBER_OF_MNIST_SAMPLES]
    else:
        inputs, targets = load_circle()

    # Initialize the correct hyperparameter tuner depending on search_type
    if search_type == 'grid_search':
        tuner = GridSearchCV(estimator, tuned_parameters)
    else:
        tuner = RandomSearchCV(estimator, tuned_parameters, n_iter=n_iter)

    # Fit the hyperparameter tuner to the data
    tuner.fit(inputs, targets)

    # Get the results and sort them by accuracy
    results = tuner.cv_results
    results = sorted(results, key=lambda x: x[1], reverse=True)

    # Return the configurations of the best experiments
    return results
