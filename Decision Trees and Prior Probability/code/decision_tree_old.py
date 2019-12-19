import numpy as np

def split_data_by_attribute(features, attribute_index, targets):
        
        feature = features[:,attribute_index].flatten()
        
        data_with_negative_feature_value = features[feature==0]
        data_with_positive_feature_value = features[feature==1]
        
        targets_with_negative_feature_value = targets[feature==0]
        targets_with_positive_feature_value = targets[feature==1]
        
        return data_with_negative_feature_value, data_with_positive_feature_value, targets_with_negative_feature_value, targets_with_positive_feature_value

class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class. 

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """
        
        
        # if no
        # run information gain on each row
        # see which attribute has the largest information gain
        # see which
        
        
        """
        MAIN QUESTION FOR ANDREAS: I don't understand how to make it iteratively produce trees down the decision tree
        I also don't understand how to make it run the decision tree to form a prediction. like how does it go from one node to the next.
        How do you find the depth of the tree and everything for the questions in free response
        * Explain what overfitting is and describe how one can tell it is happening.
        * What is the bias of the ID3 algorithm in the way it searches the hypothesis space of possible decision trees?
        do all the questions have answers from the assigned reading?
        """
        
        self._check_input(features)

        # Creating the root
        self.tree = Tree()
        
        all_features_original = np.asarray(self.attribute_names)
        
        self._extend_tree(features, targets, all_features_original, all_features_original, self.tree)

        """
        while unused_features:
            gains = []
            for attribute_name,attribute_index in zip(unused_features_original,range(len(unused_features_original))):
                
                gains.append(information_gain(features, attribute_index, targets))
                
            max_gain_index, max_gain_value = max(gains)
            
            if np.count_nonzero(gains) > 0:
                
                attr_name = unused_features_original[max_gain_index]
                attr_index = attribute_names.element(attr_name)
                
                unused_features_original.remove(attr_name)
                
                positive = Tree(value = 1, attribute_name = attr_name, attribute_index = attr_index)
                negative = Tree(value = 0, attribute_name = attr_name, attribute_index = attr_index)
                
                self.branches.append(positive)
                self.branches.append(negative)
        """
        
        
    """    
    def split_data_by_attribute(features, attribute_index, targets):
        
        feature = features[:,attribute_index]
        
        data_with_negative_feature_value = features[feature==0]
        data_with_positive_feature_value = features[feature==1]
        
        targets_with_negative_feature_value = targets[feature==0]
        targets_with_positive_feature_value = targets[feature==1]
        
        return data_with_negative_feature_value, data_with_positive_feature_value, targets_with_negative_feature_value, targets_with_positive_feature_value
    """
    
    def _extend_tree(self ,features, targets, all_features, unused_features, current_node):
        
        # list of information gains from splitting the data across all unused features
        gains = []
        
        
        # loops through list of names of unused features if any remain
        if len(unused_features) == 0 or len(np.unique(targets)) == 1:
            class_counts = np.bincount(targets)
            current_node.value = np.argmax(class_counts)
            
        else:
            for attribute_name in unused_features:
                
                # finds the index corresponding to the attribute from THE ORIGINAL list of ALL features
                attribute_idx = np.where(all_features==attribute_name)
                
                # appends the information gain of that attribute to list of all IGs
                gains.append(information_gain(features, attribute_idx, targets))
            
            
            # If there is at least one unused attribute we can split across to gain any information, we use it
            if np.count_nonzero(gains) > 0:
                
                # We identify the value of that gain (not used) and its index IN THE LIST OF UNUSED FEATURES, NOT ALONG ORIGINAL ATTRIBUTE NAMES
                max_gain_value = max(gains)
                max_gain_index = gains.index(max_gain_value)
                
                # to find index in original list of attribute names, we first find its name
                best_attr_name = unused_features[max_gain_index]
                
                print("best attribute",best_attr_name[0])
                
                # and use the name to find the index in the original list of all features
                best_attr_index = np.where(all_features==best_attr_name)
                
                # changes the list of unused names to represent the fact that this attribute has been used
                unused_feature_names_modified = np.delete(unused_features, max_gain_index)
                
                # changes the attribute index and name of the node we are currently on because it is split along this attribute
                current_node.attribute_index = best_attr_index
                current_node.attribute_name = best_attr_name
                
                # split the data along the attribute chosen to gain information as specified
                data_with_negative_feature_value, data_with_positive_feature_value, targets_with_negative_feature_value, targets_with_positive_feature_value = split_data_by_attribute(features, best_attr_index, targets)
                
                # define two branches for this current node, one for positive values of this attribute and one for negative ones
                negative = Tree()
                positive = Tree()
                
                current_node.branches.append(negative)
                current_node.branches.append(positive)
    
                # implements recursion so that we keep going down the tree except this time with the trimmed data        
                self._extend_tree(data_with_negative_feature_value, targets_with_negative_feature_value, all_features, unused_feature_names_modified, negative)
                self._extend_tree(data_with_positive_feature_value, targets_with_positive_feature_value, all_features, unused_feature_names_modified, positive)
                #                (features,                         targets,                             all_features, unused_features,               current_node)
            # if there is nothing we could separate the data by to gain information, we find what the most abundant classification is in the split data and use it for predicitons
            else:
                class_counts = np.bincount(targets)
                current_node.value = np.argmax(class_counts)
                
                print("current node value", current_node.value)
                        

        
    
    
    def predict(self, features):
        
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """
        
        self._check_input(features)
        
        predictions = []
        
        for datapoint in features:
            current_node = self.tree
            
            # if the current node has branches, we compare to their values
            while len(current_node.branches) != 0:
                # the current node has an attribute index for the feature it is split along
                # compare value of feature in datapoint to that of branches
                # if the feature is negative in datapoint, we set the current node to be that of the negative feature value
                if datapoint[current_node.attribute_index] == 0:
                    current_node = current_node.branches[0]
                    
                # vice versa for positive feature values in datapoint
                else:
                    current_node = current_node.branches[1]
            
            # we exit the while loop when we get to a leaf that has no branches
            # our prediction is the node value at that leaf so we append it to the list of predictions
            predictions.append(current_node.value)
            
        print(predictions)
            
        return np.array(predictions)
 

    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)


def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """

    
    possible_feature_values = [0,1]
    
    possible_classifications = [0,1]
    
    feature = features[:,attribute_index]
    
    
    number_of_samples = len(feature)
    
    import math
    
    
    #current_entropy = np.sum([-(len(targets[targets==possible_classification])/number_of_samples)*math.log(len(targets[targets==possible_classification])/number_of_samples, 2) for possible_classification in possible_classifications])
    
    terms_to_be_summed_for_current_entropy = []
    
    for classification in possible_classifications:
        
        number_of_elements_with_this_classification = len(targets[targets==classification])
        
        p_for_this_classification = number_of_elements_with_this_classification/len(targets)
        
        if p_for_this_classification != 0:
            terms_to_be_summed_for_current_entropy.append(-p_for_this_classification*math.log(p_for_this_classification,2))
        else:
            terms_to_be_summed_for_current_entropy.append(0)
    
    current_entropy = np.sum(terms_to_be_summed_for_current_entropy)
    
    
    
    terms_to_be_summed_for_weighted_entropy = []
    
    for possible_value in possible_feature_values:
        
        targets_split_by_feature_value = targets[feature.flatten() == possible_value]
        
        if len(targets_split_by_feature_value) != 0:
            
            """
            print("length of targets split by feature value")
            print(len(targets_split_by_feature_value))
            print("len of targets")
            print(len(targets))
            """
            
            weight_of_feature_value = len(targets_split_by_feature_value)/len(targets)
            
            terms_for_entropy_within_subset = []
            
            for classification in possible_classifications:
                
                """
                print("number of subset elements with this classification")
                print(len(targets_split_by_feature_value))
                """
                number_of_subset_elements_with_this_classification = len(targets_split_by_feature_value[targets_split_by_feature_value==classification])
                
                p_in_subset_for_this_classification = number_of_subset_elements_with_this_classification/len(targets_split_by_feature_value)
                
                if p_in_subset_for_this_classification != 0:
                    terms_for_entropy_within_subset.append(-p_in_subset_for_this_classification*math.log(p_in_subset_for_this_classification,2))
                else:
                    terms_for_entropy_within_subset.append(0)
        
            entropy_within_subset = np.sum(terms_for_entropy_within_subset)
            
            terms_to_be_summed_for_weighted_entropy.append(weight_of_feature_value*entropy_within_subset)
    
    weighted_entropy = np.sum(terms_to_be_summed_for_weighted_entropy)
    
    
    #current_entropy = np.sum(terms_to_be_summed_for_current_entropy)
        
    #weighted_entropy = np.sum([(len(feature[feature==possible_value])/number_of_samples)*(len(targets[feature==possible_value][targets[feature==possible_value]==possible_classification])/len(targets[feature==possible_value]))*math.log((len(targets[feature==possible_value][targets[feature==possible_value]==possible_classification])/len(targets[feature==possible_value])), 2) for possible_classification in possible_classifications for possible_value in possible_feature_values])

    information_gain = current_entropy - weighted_entropy    
    
    return information_gain

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
