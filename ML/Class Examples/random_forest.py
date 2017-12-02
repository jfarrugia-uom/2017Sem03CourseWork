'''
ICS5110 Machine Learning
Practical for "Evaluation, Imbalanced Data Sets, Bagging, Boosting, Random Forests"
Requires "house-votes-1984_train.txt" and "house-votes-1984_test.txt" (https://github.com/j2kun/decision-trees/blob/master/house-votes-1984.txt)
Works on both Python 2 and Python 3 (but gives different accuracy results when run using the two versions due to random functions working differently).

Add code at the end of this script to make it use decision trees in a random forest. Try to find a way to beat the score of the single decision tree. You are free to construct the forest in any way you can think of. At the end we will compare the evaluations returned.
'''

import collections
import math
import random

random.seed(0) #Set seed of the random functions to make the program deterministic.

########################################################################
class LeafNode(object):
    '''Leaf node of the decision tree that returns a class.'''
    
    def __init__(self, class_):
        '''Expects class to return for classification.'''
        self.class_ = class_
    
    def evaluate(self, input_items):
        '''Expects the input to classify.'''
        return self.class_
        
    def pretty_print(self, indentation=0):
        '''Displays the node in human friendly form.'''
        print('{}RETURN {}'.format(' '*indentation, self.class_))

########################################################################
class InternalNode(object):
    '''Internal node of the decision tree that takes a single item from the input and passes control to a child node.'''
    
    def __init__(self, input_index, value_to_child_dict):
        '''Expects the index of the input vector item to consider and a dictionary of children to pass control to given the possible values of the considered input item.'''
        self.input_index = input_index
        self.value_to_child_dict = value_to_child_dict

    def evaluate(self, input_items):
        '''Expects the input to classify.'''
        value = input_items[self.input_index]
        child = self.value_to_child_dict[value]
        return child.evaluate(input_items)
        
    def pretty_print(self, indentation=0):
        '''Displays the node in human friendly form.'''
        for (value, child) in self.value_to_child_dict.items():
            print('{}IF input_items[{}] == {}:'.format(' '*indentation, self.input_index, value))
            child.pretty_print(indentation+2)

########################################################################
def get_most_frequent_item(items):
    '''Get the most frequent item in a list (also known as the mode).'''
    freqs = collections.Counter(items) #This gives a dictionary mapping distinct items to their frequencies.

    max_freq = None
    max_item = None
    for (item, freq) in freqs.items():
        if max_freq is None or freq > max_freq:
            max_freq = freq
            max_item = item
    return max_item

########################################################################
def get_frequencies_entropy(items):
    '''Get the entropy of a list based on the frequency of its items. Returned item is a measure of purity in the list such that the minimum value returned (0) is when the list only contains one repeated item (pure) whilst the maximum is when the list contains an equal number of each possible different item (completely impure).'''
    freqs = collections.Counter(items) #This gives a dictionary mapping distinct items to their frequencies.
    total_freq = sum(freqs.values())
    
    entropy = 0.0
    for (item, freq) in freqs.items():
        proportion = float(freq)/total_freq
        entropy += -proportion*math.log(proportion)/math.log(2) #-p*log_2(p)
    return entropy

########################################################################
def get_information_gain(all_items, splits):
    '''Get the gain in frequencies entropy when a list is broken into a number of sublists. Expects the full list (all_items) and a list of sublists (splits) where all the items in the sublists came from the full list and every item in the full list is in one of the sublists.'''
    entropy_all = get_frequencies_entropy(all_items)
    
    weighted_sum_entropy_splits = 0.0
    for split in splits:
        weighted_sum_entropy_splits += float(len(split))/len(all_items) * get_frequencies_entropy(split)
    
    information_gain = entropy_all - weighted_sum_entropy_splits
    return information_gain

########################################################################
def get_splits(training_data, input_index, values):
    '''Split a training set into sublists according to a particular item position in the inputs such that each sublist has the same item at that position. Expects a training set (training_data) which is a list of tuples [(inputs, target)], an index (input_index) in the input vectors of the training set to use when splitting, and the full list of possible values that the input position should have. Returns a tuple consisting of the split training set and the same splits but with only the targets of the training set.'''
    full_splits   = { value: [] for value in values }
    target_splits = { value: [] for value in values }
    for (input_items, target) in training_data:
        value = input_items[input_index]
        full_splits[value].append( (input_items, target) )
        target_splits[value].append(target)
    return (full_splits, target_splits)

########################################################################
def id3(training_data, index_to_values):
    '''The ID3 algorithm for creating a decision tree from a training set. Expects a training set (training_data) which is a list of tuples [(inputs, target)] and a dictionary (index_to_values) mapping indexes in the input vectors of the training set to a list of all the possible values that can reside in that position. Returns a decision tree which is a LeafNode or an InternalNode. Note that you can make this algorithm consider only a subset of the input items by leaving out some indexes in index_to_values.'''
    all_targets = []
    for (input_items, target) in training_data:
        all_targets.append(target)
    most_frequent_target = get_most_frequent_item(all_targets)
    
    if all( all_targets[i] == all_targets[0] for i in range(1, len(all_targets)) ): #Check if the training set consists of the same target class throughout by checking if every item is equal to the first.
        return LeafNode(all_targets[0])
    elif len(index_to_values) == 0:
        return LeafNode(most_frequent_target)
    else:
        best_gain = None
        best_index = None
        best_full_splits = None
        for (input_index, values) in index_to_values.items():
            (full_splits, target_splits) = get_splits(training_data, input_index, values)
            gain = get_information_gain(all_targets, target_splits.values())
            if best_gain is None or gain > best_gain:
                best_gain  = gain
                best_index = input_index
                best_full_splits = full_splits

        #Create a copy of index_to_values in order to remove the item index that is used to create the current node.
        new_index_to_values = dict(index_to_values)
        new_index_to_values.pop(best_index)
        
        value_to_child_dict = dict()
        for (value, split) in best_full_splits.items():
            if len(split) == 0:
                value_to_child_dict[value] = LeafNode(most_frequent_target)
            else:
                value_to_child_dict[value] = id3(split, new_index_to_values)
        return InternalNode(best_index, value_to_child_dict)

########################################################################

#Load the dataset which is split into a training and a testing set using an 80%/20% split (348/87).
#The dataset consists of the target class ('R' or 'D') and 16 input items (all of which can be 'y', 'n', or '?'), separated by commas.
with open('house-votes-1984_train.txt', 'r') as f:
    rows = [ line.split(',') for line in f.read().split('\n')[:-1] ]
    training_data = [ (row[1:], row[0]) for row in rows ]
with open('house-votes-1984_test.txt', 'r') as f:
    rows = [ line.split(',') for line in f.read().split('\n')[:-1] ]
    testing_data  = [ (row[1:], row[0]) for row in rows ]

index_to_values = { i: [ 'y', 'n', '?' ] for i in range(16) } #All the 16 input items in the dataset have values that are 'y', 'n', or '?'.

########################################################################

#Create a single decision tree and evaluate its accuracy on the testing set.

tree = id3(training_data, index_to_values)

correct = 0
for (input_items, target) in testing_data:
    output = tree.evaluate(input_items)
    if output == target:
        correct += 1
print('Tree accuracy: {:.2%}'.format(float(correct)/len(testing_data)))

########################################################################

#Create a forest of decision trees and evaluate its accuracy on the testing set.

def get_bag(sample_size):
    '''This will give you a random subset of training set to use for the 'training_data' parameter of the 'id3' function which will allow you to use BAGGING. Just pass in the number of training set items to keep from the 16 available items and it will return the parameter to use in the 'id3' function.'''
    return random.sample(training_data, sample_size)
    
def get_random_subspace(sample_size):
    '''This will give you a random subset of indexes to use for the 'index_to_values' parameter of the 'id3' function which will allow you to use random subspace method. Just pass in the number of input items to keep from the 348 available items and it will return the parameter to use in the 'id3' function.'''
    return dict(random.sample(index_to_values.items(), sample_size))

forest = [] #Put each individual decision tree in this list.

'''------------------------
<YOUR CODE HERE>
example: forest.append(id3(get_bag(2), get_random_subspace(2)))
------------------------'''

correct = 0
for (input_items, target) in testing_data:
    outputs = [ tree.evaluate(input_items) for tree in forest ]
    output = get_most_frequent_item(outputs) #Ensemble by picking the most frequent class returned by all the decision trees.
    if output == target:
        correct += 1
print('Forest accuracy: {:.2%}'.format(float(correct)/len(testing_data)))
