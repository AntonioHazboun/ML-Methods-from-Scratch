
from sklearn.model_selection import cross_val_score

def run_experiment(estimator, params, input_data, target_data) :
    '''
    runs cross validation experiment with given estimator and input and target dataset.
    uses cross_val_score function specified from sklearn
    
    returns mean accuracy and standarddeviation of estimator accuracy in cross validation experiment
    with associated parameters.
    '''
    
    import statistics
    
    estimator.fit(input_data,target_data)
    scores = cross_val_score(estimator, X=input_data, y=target_data, cv=20)
    
    return (params, statistics.mean(scores), statistics.stdev(scores))
    
    