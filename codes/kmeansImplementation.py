import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row

def closestCluster(p, centers):
	""" This method will return the closest point index 
	for a given point P from list of points given in centers """
	bestIndex = 0
        closest = float("+inf")
        for i in range(len(centers)):
            distance = np.sqrt(np.sum((np.array(p) - centers[i]) ** 2))
            if distance < closest:
                closest = distance
                bestIndex = i
	return bestIndex

def closestClusterAndDistance(p, centers):
	""" This method will return the closest point index and its ditance 
	for a given point P from list of points given in centers """
	bestIndex = 0
        closest = float("+inf")
        for i in range(len(centers)):
            distance = np.sqrt(np.sum((np.array(p) - centers[i]) ** 2))
            if distance < closest:
                closest = distance
                bestIndex = i
        return (bestIndex, closest)

class KMeansImplemented(object):
	""" a kMeans implementation with L2 distance """

  	def __init__(self, k=3, epsilon=1e-4, maxNoOfIteration = 100):
		#self.data = data    		
		self.k = k
		self.epsilon = epsilon
		self.centers = None
		self.maxNoOfIteration = maxNoOfIteration

	def train(self, data):
		""" train method will take dataframe as an input and returns centers using kmeans algorithm with L2 distance"""
		epsilon = self.epsilon
		tempDist = 1.0
		k = self.k
		centers = data.rdd.takeSample(False, k, 1)
		i = 0 
		while tempDist > epsilon or self.maxNoOfIteration > i:
			i+=1			
			closest = data.map(lambda p: (closestCluster(p, centers), (np.array(p), 1)))
       			pointStats = closest.reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1]))
        		newPoints = pointStats.map(lambda x: (x[0], x[1][0] / float(x[1][1]))).collect()
        		tempDist = sum(np.sum((centers[index] - p) ** 2) for (index, p) in newPoints)
        		for (ind, p) in newPoints:
				centers[ind] = p
		self.centers = centers
		return self.centers

	def predict(self, data):
		""" returns the index of the closest cluster for a given data point data """
		return closestCluster(data, self.centers)
		
	def clusterAndDistance(self, data):
		""" returns the index of the closest cluster and its distance for a given data point data """
		return closestClusterAndDistance(data, self.centers)
