from multiprocessing.dummy import Pool as ThreadPool

import warnings


class Parallelizer:

    def __init__(self, function):
        '''
        Initializes parallelizer function that runs multiple runs of a function in parallel to save time
        
        args:
            param function: The function that you want to parallelize.
        '''
        self.function = function

    def parallelize(self, parameters):
        '''
        runs the function with as many parameter combinations as are listed and returns list of the return 
        value of each function running
        
        args:
            parameters: list of parameters to run with each instance of function running in parallel

        '''

        warnings.simplefilter("ignore")

        # First we set the number of threads we want start
        # You can increase/decrease that number if you want
        n_threads = 16

        # Secondly, we need a list to store the results
        results = []

        # This creates a new pool for threads
        # Whenever there is space for new threads and there are new function calls in the queue
        # The Threadpool will call the function on a new thread
        pool = ThreadPool(n_threads)

        # Here is where the parallelization happens!
        # As already described, starmap will start a new Thread as soon as there is space in teh pool
        results += pool.starmap(self.function, parameters)

        return results
