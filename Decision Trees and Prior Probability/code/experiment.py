from .decision_tree import DecisionTree
from .prior_probability import PriorProbability
from .metrics import precision_and_recall, confusion_matrix, f1_measure, accuracy
from .data import load_data, train_test_split

def run(data_path, learner_type, fraction):
    """
        1. takes in a path to a dataset and loads dataset
        2. initiates data to specified learner (prior probability or decision tree)
        3. splits data into training and testing set with training fraction involved
        4. fits specified learner to split training set
        5. tests predicted values of learner on test set
        6. returns precision_and_recall, confusion_matrix, and f1_measure from defined functions
"""
    # 1. takes in a path to a dataset
    # 2. loads it into a numpy array with `load_data`
    features, targets, attribute_names = load_data(data_path)
    
    
    # 3. instantiates the class used for learning from the data using learner_type (e.g
    #   learner_type is 'decision_tree', 'prior_probability')

    
    if learner_type == 'decision_tree':
        learner = DecisionTree(attribute_names)
    elif learner_type == 'prior_probability':
        learner = PriorProbability()
    
    
    # 4. splits the data into training and testing with `train_test_split` and `fraction`.
    train_features, train_targets, test_features, test_targets = train_test_split(features, targets, fraction)
    
    # 5. trains a learner using the training split with `fit`
    learner.fit(train_features, train_targets)
    
    if learner_type == 'decision_tree':
        learner.visualize()
    
    # 6. tests the trained learner using the testing split with `predict`
    predictions = learner.predict(test_features)
    
    # 7. evaluates the trained learner with precision_and_recall, confusion_matrix, and f1_measure
    CM = confusion_matrix(test_targets, predictions)
    acc = accuracy(test_targets, predictions)
    prec, rec = precision_and_recall(test_targets, predictions)
    f1_score = f1_measure(test_targets, predictions)


    # Order of these returns must be maintained
    return CM, acc, prec, rec, f1_score
