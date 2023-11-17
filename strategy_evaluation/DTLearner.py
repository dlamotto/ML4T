import numpy as np


class DTLearner(object):

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        return 'dlamotto3'

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
       :param data_x: A set of feature values used to train the learner
       :type data_x: numpy.ndarray
       :param data_y: The value we are attempting to predict given the X data
       :type data_y: numpy.ndarray
       """
        self.tree = self.build_tree(data_x, data_y)

    def query(self, data_x):
        """
        Estimate a set of test points given the model we built.
        :param data_x: A numpy array with each row corresponding to a specific query.
        :type data_x: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """
        predictions = np.empty(data_x.shape[0])  # initialize array to be the same size as data_x
        for i, point in enumerate(data_x):  # gives us the index and the point for each value in data_x
            # we call search_by_point to get the prediction for that point and store it in its corresponding index
            predictions[i] = self.search_by_point(self.tree, point)
        return predictions

    def search_by_point(self, tree, point):
        """
        Traverses the tree for a single feature and returns the predicted value for that point based on the tree structure
        :param tree: A numpy array with each row corresponding to a specific query.
        :type tree: numpy.ndarray
        :param point: A single point in the feature column.
        :type point: double
        :return: The predicted value for that point based on the tree structure.
        :rtype: double
        """
        feature_index, split_val, left_index, right_index = tree[0]  # extract info from the root of the subtree
        # check if the feature we are looking at is a leaf node
        if feature_index == -1:
            return split_val

        # if not a leaf node, checks the value of the feature
        if point[int(feature_index)] <= split_val:  # If the value is <= split_val, recursively call to the left tree
            return self.search_by_point(tree[int(left_index):], point)
        else:  # else, recursively call to the right tree
            return self.search_by_point(tree[int(right_index):], point)


    def get_best_feature(self, data_x, data_y):
        """
        The best feature to split on is the feature (Xi) that has the highest absolute value correlation with Y.
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        :return: The best feature's index
        :rtype: integer
        """

        best_feature_index = 0
        best_correlation = -1
        for i in range(data_x.shape[1]):
            correlation = np.corrcoef(data_x[:, i], data_y)[0, 1]
            if abs(correlation) == 1.0:
                continue  # Skip the feature if it's perfectly correlated
            if correlation > best_correlation:
                best_correlation = correlation
                best_feature_index = i

        return best_feature_index  # returns the index of the largest correlation, the best feature

    def build_tree(self, data_x, data_y):
        """
        A recursive function that constructs a decision tree as a 2d numpy array
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        :return: The decision tree
        :rtype: numpy.ndarray
        """

        # aggregated all the data left into a leaf if leaf_size or fewer entries left
        if data_x.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data_y), -1, -1]])

        i = self.get_best_feature(data_x, data_y)
        split_val = np.median(data_x[:, i])

        if np.unique(data_x[:, i]).shape[0] == 1:
            return np.array([[-1, np.mean(data_y), -1, -1]])

        left_indices = data_x[:, i] <= split_val  # stores the values <= to the median value
        right_indices = data_x[:, i] > split_val  # stores the values > to the median value

        if not np.any(left_indices) or not np.any(right_indices):
            return np.array([[-1, np.mean(data_y), -1, -1]])

        left_tree = self.build_tree(data_x[left_indices], data_y[left_indices])
        # if np.all(np.isclose(left_tree, left_tree[0])):
        #     return np.array([[-1, np.mean(data_y), -1, -1]])
        right_tree = self.build_tree(data_x[right_indices], data_y[right_indices])

        if left_tree.ndim == 1:
            root = np.asarray([i, split_val, 1, 2])
        else:
            # creates the root of the current subtree
            root = np.asarray([i, split_val, 1, left_tree.shape[0] + 1])

        return np.row_stack((root, left_tree, right_tree))
