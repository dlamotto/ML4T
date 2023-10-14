import numpy as np
class RTLearner:
    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor:
        leaf_size: the maximum number of samples to be aggregated at a leaf (aka max # of samples reqired to split
        an internal node)
        verbose: If True, generate output to a screen for debugging purposes.
        """
        self.leaf_size = leaf_size
        self.verbose = verbose

    def author(self):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
        """
        return "dlamotto3"

    def is_leaf(self,  data_x, data_y):
        """
        If data_x contains only one data point or if all values in data_y are the same, should be treated as a leaf.
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        :return: True if the node is a leaf, false otherwise
        :rtype: Boolean
        """
        if data_x.shape[0] == 1 or np.all(data_y == data_y[0]):
            return True
        else:
            return False

    def get_random_feature(self, data_x):
        """
        TSelect a random feature for splitting
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :return: The random feature's index
        :rtype: integer
        """
        return np.random.randint(data_x.shape[1])  # Randomly select a feature index

    def build_random_tree(self, data_x, data_y):
        """
        A recursive function that constructs a random tree as a 2d numpy array
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        :return: The random tree
        :rtype: numpy.ndarray
        """
        # -1 indicates a leaf node
        if data_x.shape[0] <= self.leaf_size:
            return np.array([[-1, np.mean(data_y), -1, -1]])

        i = self.get_random_feature(data_x)  # Get random feature index

        # Compute the split value by taking the median of a random feature
        split_val = np.median(data_x[:, i])

        if np.unique(data_x[:, i]).shape[0] == 1:
            return np.array([[-1, np.mean(data_y), -1, -1]])

        # Splitting the data
        left_indices = data_x[:, i] <= split_val  # stores the values <= to the median value
        right_indices = data_x[:, i] > split_val  # stores the values > to the median value

        if not np.any(left_indices) or not np.any(right_indices):
            return np.array([[-1, np.mean(data_y), -1, -1]])

        left_tree = self.build_random_tree(data_x[left_indices], data_y[left_indices])
        right_tree = self.build_random_tree(data_x[right_indices], data_y[right_indices])

        # creates the root of the current subtree
        root = np.array([[i, split_val, 1, left_tree.shape[0] + 1]])

        if left_tree.ndim == 1:
            root = np.asarray([i, split_val, 1, 2])
        else:
            # creates the root of the current subtree
            root = np.asarray([i, split_val, 1, left_tree.shape[0] + 1])

        return np.row_stack((root, left_tree, right_tree))

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        self.tree = self.build_random_tree(data_x, data_y)

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