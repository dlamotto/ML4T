import numpy as np

class BagLearner:
    def __init__(self, learner, kwargs={}, bags=1, boost=False, verbose=False):
        """
        Constructor:
        learner: Points to the learning class that will be used in the BagLearner.
        kwargs: Keyword arguments that are passed on to the learner’s constructor and can vary.
        bags: Number of (bags) learners you should train using Bootstrap Aggregation
        boost: If true, then you should implement boosting (optional implementation)
        verbose: If True, generate output to a screen for debugging purposes.
        """
        self.learners = []
        self.kwargs = kwargs
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

        # create 'bags' number of learners with the specified 'kwargs' and store tham in the learners list
        for _ in range(self.bags):
            self.learners.append(learner(**kwargs))

    def author(self):
        """  		  	   		  		 		  		  		    	 		 		   		 		  
        :return: The GT username of the student  		  	   		  		 		  		  		    	 		 		   		 		  
        :rtype: str  		  	   		  		 		  		  		    	 		 		   		 		  
        """
        return "dlamotto3"

    def add_evidence(self, data_x, data_y):
        """
        Train the bagged learners with the provided training data.
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict given the X data
        :type data_y: numpy.ndarray
        """
        n = data_x.shape[0]  # determines the number of rows in data_x

        for learner in self.learners:
            # bootstrap samples; random samples of the training data selected with replacement
            indices = np.random.choice(n, size=n, replace=True)
            # extract the corresponding rows from data_x and data_y using the randomly selected indices
            bootstrap_x = data_x[indices]
            bootstrap_y = data_y[indices]
            learner.add_evidence(bootstrap_x, bootstrap_y)  # train each learner on a different bootstrap sample

    def query(self, data_x):
        """
        Predict outcomes based on the trained bagged learners.
        :param data_x: A numpy array with each row corresponding to a specific query.
        :type data_x: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """

        all_predictions = []
        # convert predictions for each learner into a 2d numpy array
        for learner in self.learners:
            single_prediction = learner.query(data_x)
            all_predictions.append(single_prediction)

        predictions = np.array(all_predictions)
        # predictions get averaged column-wise to get a single prediction value

        return predictions.mean(axis=0)

