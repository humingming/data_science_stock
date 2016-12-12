"""
Test BagLearner
"""

import numpy as np
import pandas as pd

import math
import KNNLearner as knn
import LinRegLearner as lrr
import BagLearner as bl
import matplotlib.pyplot as plt

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

    # Create BagLearner, which uses KNN and time execution.
    start = time.time()
    learner = bl.BagLearner(learner = knn.KNNLearner, kwargs={"k":4}, bags=4)
    learner.add_evidence(trainX, trainY)
    
    # Evaluate in sample
    predY = learner.query(trainX)
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print('\nKNN in sample results') 
    print('RMSE: ', rmse)
    c = np.corrcoef(predY, y=trainY)
    print('corr:', c[0,1])
    
    # Evaluate out of sample
    predY = learner.query(testX)
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print('KNN out of sample results')
    print('RMSE: ', rmse)
    c = np.corrcoef(predY, y=testY)
    print('corr: ', c[0,1])
    end = time.time()
    print('Time:', end - start)

    # Create BagLearner, which uses linear regression learner.
    learner = bl.BagLearner(learner = lrr.LinRegLearner, kwargs={}, bags=20)
    learner.add_evidence(trainX, trainY)
    
    # Evaluate in sample
    predY = learner.query(trainX)
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print('\nLinReg in sample results') 
    print('RMSE: ', rmse)
    c = np.corrcoef(predY, y=trainY)
    print('corr:', c[0,1])
    
    # Evaluate out of sample
    predY = learner.query(testX)
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print('LinReg out of sample results')
    print('RMSE: ', rmse)
    c = np.corrcoef(predY, y=testY)
    print('corr: ', c[0,1])

    plt.clf()
    plt.plot(K,rms_rf_out, K, rms_rf_in)
    plt.legend(['RMSE Bag Out', 'RMSE Bag In'])
    plt.ylabel("Root Mean Square Error")
    plt.xlabel("K")



