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
        Binary decision tree class with multiple branches at each node. If a node has no branches 
        then it is assigned a value corresponding to the class it predicts
        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of classes used to traverse tree, first branch is negative 
            attibute value, second is positive attribute value. If node is leaf node, branches
            list length is zero.
            attribute_name (str): name of attribute upon which data is split
            attribute_index (float): attribute name index
            value (number): contains reference attribute value
        """
        
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        Binary decision tree based on ID3 algorithm. Consists of nested Tree classes
        
        Args:
            attribute_names (list): list of attribute names 
        """
        self.attribute_names = attribute_names
        self.tree = None

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match"
            )

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxK containing the K features
            targets (np.array): numpy array of size Nx1 containing the 1 feature.
        """
        
        
        # if no
        # run information gain on each row
        # see which attribute has the largest information gain
        # split on this attribute
        
        
        self._check_input(features)

        # Creating the root
        self.tree = Tree()
        
        all_features_original = np.asarray(self.attribute_names)
        
        self._extend_tree(features, targets, all_features_original, all_features_original, self.tree)

        
    
    def _extend_tree(self ,features, targets, all_features, unused_features, current_node):
        
        # list of information gains from splitting the data across all unused features
        
        # loops through list of names of unused features if any remain
        
        if len(unused_features) == 0 or len(np.unique(targets)) == 1:
            
            negative_and_positive_count = [0,0]
            for target in targets:
                negative_and_positive_count[target] += 1 
            
            if negative_and_positive_count[0] < negative_and_positive_count[1]:
                current_node.value = 1
            elif negative_and_positive_count[0] > negative_and_positive_count[1]:       
                current_node.value = 0
            
            
        else:
            gains = []
            print("number of unused features", len(unused_features))
            
            for attribute_name, idx in zip(unused_features,range(len(unused_features))):
                
                # finds the index corresponding to the attribute from THE ORIGINAL list of ALL features
                attribute_idx = np.where(all_features==attribute_name)
                
                # appends the information gain of that attribute to list of all IGs
                gains.append(information_gain(features, idx, targets))
            
            
            # If there is at least one unused attribute we can split across to gain any information, we use it
            
            # We identify the value of that gain (not used) and its index IN THE LIST OF UNUSED FEATURES, NOT ALONG ORIGINAL ATTRIBUTE NAMES
            max_gain_value = max(gains)
            max_gain_index = gains.index(max_gain_value)
            
            # to find index in original list of attribute names, we first find its name
            best_attr_name = unused_features[max_gain_index]
            
            
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

            # len(targets_with_negative_feature_value) == 0 or 
            if isinstance(targets_with_negative_feature_value is not None:
                if len(np.unique(targets_with_negative_feature_value)) == 1:
                    negative.value(targets_with_negative_feature_value[0]) # any would do, just picked first one
                else:
                    # implements recursion so that we keep going down the tree except this time with the trimmed data       
                    self._extend_tree(data_with_negative_feature_value, targets_with_negative_feature_value, all_features, unused_feature_names_modified, negative)
            
            if targets_with_positive_feature_value is not None:
                if len(np.unique(targets_with_positive_feature_value)) == 1:
                    positive.value(targets_with_positive_feature_value[0]) # any would do, just picked first one
                else:
                    self._extend_tree(data_with_positive_feature_value, targets_with_positive_feature_value, all_features, unused_feature_names_modified, positive)
                    #                (features,                         targets,                             all_features, unused_features,               current_node)
        
                        

        
    
    
    def predict(self, features):
        
        """
        Takes in features and predicts classes for each point using trained model

        Args:
            features (np.array): numpy array of size NxK containing the K features
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

            
        return np.array(predictions)
 
    

def information_gain(features, attribute_index, targets):
    """
    returns information gain from classic definition of entropy loss from wiki page 
    after splitting data along attribute of attribute_index
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
            
            
            weight_of_feature_value = len(targets_split_by_feature_value)/len(targets)
            
            terms_for_entropy_within_subset = []
            
            for classification in possible_classifications:
                
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
