
import itertools
from copy import deepcopy

from .parallelizer import Parallelizer
from .worker import run_experiment


class GridSearchCV:

    def __init__(self, estimator, tuned_parameters):
        '''
        Generates every possible parameter combination for the support vector machine and tries them out
        to identify the best one
        '''

        # This variable should contain the final results after fitting your dataset in an order manner
        self.cv_results = []
        self.estimator = estimator
        self.tuned_parameters = tuned_parameters

    def fit(self, inputs, targets):
        '''                                                                                                                
        Creates multiple SVMs with different parameter combinations, running them in parallel
        and logging the results of each combination on the input and target combinations
        
        args:
            inputs: np array of inputs
            targets: np array of classifications of input points
        '''

        self.combinations_list = self.generate_all_permutations(self.tuned_parameters)
        
        list_of_estimators_to_run = []
        
        for combination_dict in self.combinations_list:
            estimator_copy = deepcopy(self.estimator)
            estimator_copy.__init__(**combination_dict)
            estimator_tuple = (estimator_copy, combination_dict, inputs, targets)
            
            list_of_estimators_to_run.append(estimator_tuple)
        
        parallelizer_object = Parallelizer(run_experiment)
        self.cv_results = parallelizer_object.parallelize(list_of_estimators_to_run)

    def generate_all_permutations(self, tuned_parameters):
        '''
        Generates list of dictionaries containing every possible input parameter combination
        for the SVM
        '''        
        
        combinations = itertools.product(*(tuned_parameters[parameter] for parameter in tuned_parameters))
        combinations_list = []
        for combination_values in combinations:
            permutation = {key:value for key,value in zip(tuned_parameters.keys(),combination_values)}
            combinations_list.append(permutation)
                
        return combinations_list

