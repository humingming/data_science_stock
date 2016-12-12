""" A wrapper for Bootstrap Aggregating. """

import numpy as np
import math
from random import randrange

class BagLearner:

    def __init__(self, learner, kwargs, bags, boost=False, verbose=False):
        """ Parameters
            ---------------
            learner - the type of learner used for bagging
            kwargs - arguments for learner
            bags - the number of "bags" (subsets of data) used for training
            boost - use Ada boost?
            verbose - print yielded Y values?
        """
        self.learner = learner(**kwargs)
        self.bags = bags
        self.boost = boost
        self.verbose = verbose
        self.dataX = None
        self.dataY = None

    def add_evidence(self, dataX, dataY):
        """ Add training data. """
        self.dataX = dataX
        self.dataY = dataY

        
    def query(self, points):
        """ Get estimated values by bagging
            For each "bag":
            1) Get some random indices (uniformly, w/ replacement).
            2) Set the learner's training data using these indices.
            3) Get the estimated Y values from the learner.
            
            Finally, return the mean of all learners' estimates.
        """
        
        def get_rand():    
            """ Generate random integers(with replacement), use to extract data at random.
                All integers are in range (0, total number of indices in dataX).
                Return a subset of the original data for X and Y using the random indices.
            """
            rands = []
            for i in range(len(self.dataX)):
                rands.append(randrange(len(self.dataX)))
            return self.dataX[rands], self.dataY[rands]
        
        learners = []
        for i in range(0, self.bags):
            data_x, data_y = get_rand()
            self.learner.add_evidence(data_x, data_y)
            est_Y = self.learner.query(points)
            learners.append(est_Y)
            
        # Return the mean of the outputs
        return np.mean(learners, axis=0)
