"""
Test KNNLearner 
"""

import numpy as np
import math
import KNNLearner as knn
import time

if __name__=="__main__":
    # Read data
    data = np.genfromtxt('Data/ripple.csv', delimiter=',')

    # compute how much of the data is training and testing
    train_rows = math.floor(0.60* data.shape[0])
    test_rows = data.shape[0] - train_rows

    # separate out training and testing data
    trainX = data[:train_rows,0:-1]
    trainY = data[:train_rows,-1]
    testX = data[train_rows:,0:-1]
    testY = data[train_rows:,-1]

    # Create KNN Learner and time execution.
    start = time.time()
    learner2 = knn.KNNLearner(3)
    learner2.add_evidence(trainX, trainY)
    
    # Evaluate in sample
    predY = learner2.query(trainX)
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print('\nIn sample results') 
    print('RMSE: ', rmse)
    c = np.corrcoef(predY, y=trainY)
    print('corr:', c[0,1])
    
    # Evaluate out of sample
    predY = learner2.query(testX)
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print('Out of sample results')
    print('RMSE: ', rmse)
    c = np.corrcoef(predY, y=testY)
    print('corr: ', c[0,1])
    end = time.time()
    print(end - start)
