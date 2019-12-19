
import random
from copy import deepcopy

from .parallelizer import Parallelizer
from .worker import run_experiment


class RandomSearchCV:

    def __init__(self, estimator, param_distributions, n_iter=5):
        '''
        generates an n_iter number of parameter combinations and puts them in a list to try out
        '''

        # This variable should contain the final results after fitting your dataset
        self.cv_results = []
        self.estimator = estimator
        self.param_distributions = param_distributions
        self.n_iter = n_iter

    def fit(self, inputs, targets):
        '''
        runs the generated random parameter combinations on different estimators evaluated on the
        input and target data. Runs these estimators in parallel and reports all results
        '''
        self.generate_random_permutations(self.param_distributions)
        
        list_of_estimators_to_run = []
        
        for combination_dict in self.combinations_list:
            estimator_copy = deepcopy(self.estimator)
            estimator_copy.__init__(**combination_dict)
            estimator_tuple = (estimator_copy, combination_dict, inputs, targets)
            
            list_of_estimators_to_run.append(estimator_tuple)
        
        parallelizer_object = Parallelizer(run_experiment)
        self.cv_results = parallelizer_object.parallelize(list_of_estimators_to_run)
        
        
    def generate_random_permutations(self, param_distributions):
        
        permutations = []
        
        for iteration in range(self.n_iter):
            permutation = []
            for key in param_distributions.keys():
                permutation.append(random.choice(param_distributions[key]))
            permutations.append(permutation)
        
        combinations_list = []
        for permuation_values in permutations:
            combination = {key:value for key,value in zip(param_distributions.keys(),permuation_values)}
            combinations_list.append(combination)
            
        self.combinations_list = combinations_list
