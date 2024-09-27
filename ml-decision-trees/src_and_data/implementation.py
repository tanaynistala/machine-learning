import numpy as np


def counting_heuristic(x_inputs, y_outputs, feature_index, classes):
    """
    Calculate the total number of correctly classified instances for a given
    feature index, using the counting heuristic.

    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param feature_index: int, index of the feature to be evaluated
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: int, total number of correctly classified instances
    """

    # Split the output based on input class
    input_classes = np.unique(x_inputs[:,feature_index])
    subsets = [ y_outputs[x_inputs[:,feature_index] == c] for c in input_classes ]

    # Get the number of correct predictions (majority outputs) in each subset
    counts  = [ np.unique(s, return_counts=True)[1].max() for s in subsets ]
    total_correct = np.sum(counts)

    return total_correct


def set_entropy(x_inputs, y_outputs, classes):
    """
    Calculate the entropy of the given input-output set.

    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: float, entropy value of the set
    """

    probs = [ np.where(y_outputs == c, 1, 0).mean() for c in classes ]
    entropy = -np.sum([ 0 if p == 0 else p * np.log2(p) for p in probs ])

    return entropy


def information_remainder(x_inputs, y_outputs, feature_index, classes):
    """
    Calculate the information remainder after splitting the input-output set based on the
    given feature index.


    :param x_inputs: numpy array of shape (n_samples, n_features), containing the input data
    :param y_outputs: numpy array of shape (n_samples,), containing the output data (class labels)
    :param feature_index: int, index of the feature to be evaluated
    :param classes: container, unique set of class labels.
           e.g. [0,1] for two classes or [0,1,2] for 3 classes
    :return: float, information remainder value
    """

    # Calculate the entropy of the overall set
    overall_entropy = set_entropy(x_inputs, y_outputs, classes)

    # Calculate the entropy of each split set
    input_classes = np.unique(x_inputs[:,feature_index])
    y_subsets = [ y_outputs[x_inputs[:,feature_index] == c] for c in input_classes ]

    set_entropies = [ set_entropy(x_inputs, s, classes) for s in y_subsets ]

    # Calculate the remainder
    remainder = np.sum([ H * len(subset) / len(y_outputs) for H, subset in zip(set_entropies, y_subsets) ])

    gain = overall_entropy - remainder

    return gain
