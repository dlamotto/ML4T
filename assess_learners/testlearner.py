""""""
from matplotlib import pyplot as plt

"""  		  	   		  		 		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
"""

import math
import sys
import time
import numpy as np

import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as it

# command line prompt:
# $env:PYTHONPATH = "..\;."; python testlearner.py Data/Istanbul.csv


def exp1(train_x, train_y, test_x, test_y):
    leaf_size = 50
    in_sample_rsmes = []
    out_sample_rsmes = []

    for each_leaf_size in range(1, leaf_size + 1):
        learner = dt.DTLearner(leaf_size=each_leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)

        # in sample
        predY_train = learner.query(train_x)
        RMSE_train = np.sqrt(np.mean((train_y - predY_train) ** 2))

        # out sample
        predY_train = learner.query(test_x)
        RMSE_test = np.sqrt(np.mean((test_y - predY_train) ** 2))

        in_sample_rsmes.append(RMSE_train)
        out_sample_rsmes.append(RMSE_test)

    x = np.array([[i for i in range(50)]])
    reshaped_arr = x.reshape(50, 1)
    plt.plot(reshaped_arr, in_sample_rsmes, label=f"In-sample RMSE")
    plt.plot(reshaped_arr, out_sample_rsmes, label=f"Out-sample RMSE")

    plt.title("Overfitting with Respect to Leaf Size in Decision Trees")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig("images/figure1.png")
    plt.clf()


def exp2(train_x, train_y, test_x, test_y):
    leaf_size = 50
    num_of_bags = 20
    in_sample_rsmes = []
    out_sample_rsmes = []

    for each_leaf_size in range(1, leaf_size + 1):
        learner_instance = bl.BagLearner(learner=dt.DTLearner, kwargs={"leaf_size": each_leaf_size}, bags=num_of_bags, boost=False,
                                verbose=False)
        learner_instance.add_evidence(train_x, train_y)

        # in sample
        predY_train = learner_instance.query(train_x)
        RMSE_train = np.sqrt(np.mean((train_y - predY_train) ** 2))

        # out sample
        predY_train = learner_instance.query(test_x)
        RMSE_test = np.sqrt(np.mean((test_y - predY_train) ** 2))

        in_sample_rsmes.append(RMSE_train)
        out_sample_rsmes.append(RMSE_test)

    x = np.array([[i for i in range(50)]])
    reshaped_arr = x.reshape(50, 1)
    plt.plot(reshaped_arr, in_sample_rsmes, label=f"In-sample RMSE")
    plt.plot(reshaped_arr, out_sample_rsmes, label=f"Out-sample RMSE")

    plt.title("Overfitting w Respect to Leaf Size "
              "in BagLearner using DT and 20 bags")
    plt.xlabel("Leaf Size")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig("images/figure2.png")
    plt.clf()


def exp3_1(train_x, train_y, test_x, test_y):
    leaf_size = 25
    out_sample_mape_dt = []
    out_sample_mape_rt = []

    for each_leaf_size in range(1, leaf_size + 1):
        learner = dt.DTLearner(leaf_size=each_leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)
        predY_test = learner.query(test_x)

        mape_value_dt = np.mean(np.abs((np.asarray(test_y) - np.asarray(predY_test)) / test_y)) * 100
        out_sample_mape_dt.append(mape_value_dt)

        learner = rt.RTLearner(leaf_size=each_leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)
        predY_test = learner.query(test_x)

        mape_value_rt = np.mean(np.abs((np.asarray(test_y) - np.asarray(predY_test)) / test_y)) * 100
        out_sample_mape_rt.append(mape_value_rt)

    # print(out_sample_mape_dt)
    # print(out_sample_mape_rt)

    x = np.array([[i for i in range(25)]])
    reshaped_arr = x.reshape(25, 1)
    plt.plot(reshaped_arr, out_sample_mape_dt, label=f"DT: Out-sample MAPE ")
    plt.plot(reshaped_arr, out_sample_mape_rt, label=f"RT: Out-sample MAPE")

    plt.title("Comparing MAPE in Decision Trees and Random Tree")
    plt.xlabel("Leaf Size")
    plt.ylabel("MAPE")
    plt.legend()
    plt.savefig("images/figure3.png")
    plt.clf()


def exp3_2(train_x, train_y, test_x, test_y):
    leaf_size = 25
    out_sample_mae_dt = []
    out_sample_mae_rt = []

    for each_leaf_size in range(1, leaf_size + 1):
        learner = dt.DTLearner(leaf_size=each_leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)
        predY_test = learner.query(test_x)
        out_sample_mae_dt_train = np.mean(np.abs((np.asarray(test_y) - np.asarray(predY_test)))) * 100
        out_sample_mae_dt.append(out_sample_mae_dt_train)

        learner = rt.RTLearner(leaf_size=each_leaf_size, verbose=False)
        learner.add_evidence(train_x, train_y)
        predY_test = learner.query(test_x)
        out_sample_mae_rt_train = np.mean(np.abs((np.asarray(test_y) - np.asarray(predY_test)))) * 100
        out_sample_mae_rt.append(out_sample_mae_rt_train)
    #
    # print(out_sample_mae_dt)
    # print(out_sample_mae_rt)

    x = np.array([[i for i in range(25)]])
    reshaped_arr = x.reshape(25, 1)
    plt.plot(reshaped_arr, out_sample_mae_dt, label=f"DT: Out-sample MAE ")
    plt.plot(reshaped_arr, out_sample_mae_rt, label=f"RT: Out-sample MAE")

    plt.title("Comparing MAE in Decision Trees and Random Tree")
    plt.xlabel("Leaf Size")
    plt.ylabel("MAE")
    plt.legend()
    plt.savefig("images/figure4.png")
    plt.clf()


def exp3_3(train_x, train_y):
    max_train_size = train_x.shape[0]
    running_time_dt = []
    running_time_rt = []

    for training_size in range(200, max_train_size + 1, 200):
        temp_trainX = train_x[:training_size]
        temp_trainY = train_y[:training_size]

        learner = dt.DTLearner(leaf_size=1, verbose=False)
        start = time.time()
        learner.add_evidence(temp_trainX, temp_trainY)
        end = time.time()
        running_time = end - start
        running_time_dt.append(running_time)

        learner = rt.RTLearner(leaf_size=1, verbose=False)
        start = time.time()
        learner.add_evidence(temp_trainX, temp_trainY)
        end = time.time()
        running_time = end - start
        running_time_rt.append(running_time)

    print(running_time_dt)
    print(running_time_rt)
    plt.plot(250, running_time_dt, label="Decision Tree")
    plt.plot(250, running_time_rt, label="Random Tree")
    plt.title("Decision Tree vs Random Tree on training time")
    plt.xlabel("Training Sizes")
    plt.ylabel("Training Times")
    plt.legend()
    plt.savefig("figure5.png")
    plt.clf()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)
    inf = open(sys.argv[1])

    option = 2
    np.random.seed(42398048)

    for i in range(5):
        if i == 0:
            learner = lrl.LinRegLearner(verbose=False)
        elif i == 1:
            learner = dt.DTLearner(verbose=False)
        elif i == 2:
            learner = rt.RTLearner(verbose=False)
        elif i == 3:
            learner = bl.BagLearner(learner=rt.RTLearner, kwargs={"leaf_size": 1}, bags=20, boost=False, verbose=False)
        elif i == 4:
            learner = it.InsaneLearner(verbose=False)

    alldata = np.array([list(map(str, s.strip().split(','))) for s in inf.readlines()])

    if sys.argv[1] == "Data/Istanbul.csv":
        alldata = alldata[1:, 1:]

    alldata = alldata.astype('float')
    np.random.shuffle(alldata)

    # compute how much of the data is training and testing
    train_rows = int(0.6 * alldata.shape[0])   # calculates the cutoff for 60% used to split the data
    test_rows = alldata.shape[0] - train_rows

    # separate out training and testing data
    train_x = alldata[:train_rows, 0:-1]
    train_y = alldata[:train_rows, -1]
    test_x = alldata[train_rows:, 0:-1]
    test_y = alldata[train_rows:, -1]

    # print(f"{test_x.shape}")
    # print(f"{test_y.shape}")

    # create a learner and train it
    learner.add_evidence(train_x, train_y)  # train it
    # print(learner.author())

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse_train = np.sqrt(np.mean((train_y - pred_y) ** 2))
    # print()
    # print("In sample results")
    # print(f"RMSE: {rmse}")
    c_train = np.corrcoef(pred_y, y=train_y)
    # print(f"corr: {c[0, 1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x) # get the predictions
    rmse_test = np.sqrt(np.mean((test_y - pred_y) ** 2))
    # print()
    # print("Out of sample results")
    # print(f"RMSE: {rmse}")
    c_test = np.corrcoef(pred_y, y=test_y)
    # print(f"corr: {c[0, 1]}")

    exp1(train_x, train_y, test_x, test_y)
    exp2(train_x, train_y, test_x, test_y)
    exp3_1(train_x, train_y, test_x, test_y)
    exp3_2(train_x, train_y, test_x, test_y)
    exp3_3(train_x, train_y)



