import BagLearner as bl
import LinRegLearner as lr
import numpy as np

class InsaneLearner:
    def __init__(self, verbose=False):
        self.learners = [bl.BagLearner(learner=lr.LinRegLearner, kwargs={}, bags=20, verbose=verbose) for _ in range(20)]

    def author(self):
        return "dlamotto3"

    def add_evidence(self, data_x, data_y):
        for learner in self.learners:
            learner.add_evidence(data_x, data_y)

    def query(self, data_x):
        all_predictions = []
        for learner in self.learners:
            single_prediction = learner.query(data_x)
            all_predictions.append(single_prediction)

        predictions = np.array(all_predictions)
        return predictions.mean(axis=0)


# constructor instantiates 20 BagLearner instances where each instance is
# composed of 20 LinRegLearner instances.

# add_evidence trains each BagLearner with the data

# query gets predictions from all the BagLearners and returns teh average of them
