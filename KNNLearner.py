"""
A wrapper for K-nearest neighbours learning.
TODO: Probably implement it using kd-tree later.
"""

import numpy as np
import math
from heapq import heappush, heapreplace, nsmallest

class KNNLearner:

    def __init__(self, k=5, verbose=False):
        """ k-nearest neighbours learner, with default value of k=3. """
        if k <= 0:
            raise ValueError('k should be > 0')
        self.k = k
        self.dataX = None
        self.dataY = None
        self.verbose = verbose

    def add_evidence(self, dataX, dataY):    
        self.dataX = dataX 
        self.dataY = dataY

    def predict_y(self, point):
        """ Predict Y value based on k nearest neighbours.
            The method uses a heapq to keep track of the k nearest neighbours.
            
            The heap queue is using the negative euclidean distance for priority.
            That way, the larger distances have smaller value (and priority).

            The heapq contains pairs (K, V), where:
            K is negated euclidean distance between query and data vectors.
            V is the Y-value of the data vector.
            --------------------------------------
            Implemented in the following way:
            I) Fill the heap with the first k vectors from data set
            II) Keep a current minimum (the vector farthest from query vector).
            III) For each of the remaining vectors in data set.
                1) Compute the distance between query and data vector.
                2) Negate the distance (larger distances have smaller value).
                3) If the current value is larger than current minimum:
                    - Remove current min from heap and add current pair.
                    - Update current min.

            Note: Size of heap is preserved (always == k) by always removing element
            with smallest priority (largest distance) before adding new one.
        """
        heap = []
        
        # Add first k elements to fill heapq
        for val in zip(self.dataX[:self.k], self.dataY[:self.k]):
            diff = point-val[0] # coordinates are at pos 0
            n_summed = np.sum((np.square(diff))) # sum the squares of diff
            euclid_dist = -math.sqrt(n_summed) # negate the distance
            pair = (euclid_dist, val[1]) # y values are at position 1
            heappush(heap, pair)

        # Compute the current minimum (largest distance).
        # Updated whenever the smallest element is removed.
        curr_min = nsmallest(1, heap)[0][0]
        
        # Check remaining elements
        for val in zip(self.dataX[self.k:], self.dataY[self.k:]): 
            diff = point-val[0]
            n_summed = (np.sum(np.square(diff)))
            euclid_dist = -math.sqrt(n_summed)
            
            if euclid_dist > curr_min:
                pair = (euclid_dist, val[1])
                heapreplace(heap, pair) # Remove smallest, add current
                curr_min = nsmallest(1, heap)[0][0] # Update current min elem
        
        # Get the Y-values from the heap and return the mean.
        result = [x[1] for x in heap]
        return sum(result) / float(len(result))

    def query(self, points):
        """ Query predicted data for each point/vector in points. """
        if self.dataX is None or self.dataY is None:
            raise UnboundLocalError(
            'Training data not found. Use addEvidence to add training data first.')
        pred = []
        for point in points:
            predY = self.predict_y(point)
            pred.append(predY)

        if self.verbose:
            print(pred)
            
        return np.asarray(pred)
