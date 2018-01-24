import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
import xml.etree.ElementTree as ET
from pyspark.sql.functions import stddev_pop, avg, broadcast
import re
from collections import Counter 
from kmeansImplementation import KMeansImplemented

def preProcessPosts(data):
	""" method will take single row of post as input in xml format 
	and returns a list with the required attributes"""
	try:
		root = ET.fromstring(data)
	        return [int(root.attrib['Id']), int(root.attrib['PostTypeId']), int(root.attrib['Score']), int(root.attrib['ViewCount']), (root.attrib['Tags']), int(root.attrib['AnswerCount']), int(root.attrib['CommentCount']), int(root.attrib['FavoriteCount'])]
	except:
		print("Ignoring record")

def helper(val, columns):
	""" Helper function used in oneHotName method to set the column as 1 for each badge of the user"""
	a = [0] * len(columns)
	val1 = val.split("><")
	val2 = [x.replace("<","").replace(">","") for x in val1]
	match = [re.sub('[^a-zA-Z0-9+#]', '_', x) for x in val2]
	for i in match:
		try:
			a[columns.index(i)] = 1
		except:
			continue
	return a

def normalizeFeatures(df, cols):
	""" Normalized feature method is used to normalize each feature passed into the list cols"""
	allCols = df.columns
	#remove the cols to normalized to set the columns of return dataframe
	_ = [allCols.remove(x) for x in cols]
	# calculate the avg and stddev of the features to normalized
	stats = (df.groupBy().agg(*([stddev_pop(x).alias(x + '_stddev') for x in cols] + [avg(x).alias(x + '_avg') for x in cols])))
	# broadcast and join into current DF 
	df = df.join(broadcast(stats))
	# normalized the columns and select the required columns gor final DF
	exprs = [x for x in allCols] + [((df[x] - df[x + '_avg']) / df[x + '_stddev']).alias(x) for x in cols]  
	return df.select(*exprs)

def error(point, clusters):
	""" find the square root of sum of square of point from its cluster center """
    	center = clusters.centers[clusters.predict(point)]
    	return np.sqrt(np.sum([x**2 for x in (point - center)]))

if __name__ == "__main__":
	sc = SparkContext(appName="kmeans_posts")
	sqlContext = SQLContext(sc)
	
	posts = sc.textFile("/data/stackoverflow/Posts", 50)
		
	# A function preProcessPosts is defined to parse Post.xml and retrieve the list of its attributes using Element Tree package	
	postsData = posts.map(preProcessPosts).filter(lambda x: x is not None)
	
	#Extracting tags from the postsData rdd
	tags = postsData.map(lambda x: x[4]).flatMap(lambda y: y.split("><"))
	
	#Filtering tags rdd from unwanted characters
	tagsFiltered1 = tags.map(lambda x: x.replace("<","").replace(">",""))
	tagsFiltered2 = tagsFiltered1.map(lambda x: re.sub('[^a-zA-Z0-9+#]', '_', x))
	
	#counting the total number of tags by .countByvalue method
	tagsCount = tagsFiltered2.countByValue()

	#taking only 100 most common distinct tags using .most_common method from Counter package to avoid computational errors
	tags100 = dict(Counter(tagsCount).most_common(100))
	
	#assigning tags values to variable columns
	columns = tags100.keys()
	
	# A final rdd is made by taking all the attributes including tags that are passed to helper function to convert to one hot notation
	processedData = postsData.map(lambda p :(p[0], p[1], p[2], p[3], p[5], p[6], p[7], helper(p[4], columns)))
	
	#Converting data to row object to create dataframe
	finalProcessedData = processedData.map(lambda (c0, c1, c2, c3, c5, c6, c7, data): Row(c0, c1, c2, c3, c5, c6, c7, *[eachColumn for eachColumn in data]))
	finalDF = finalProcessedData.toDF(['PostId', 'PostTypeId', 'Score', 'ViewCount', 'AnswerCount', 'CommentCount', 'FavoriteCount'] + columns).cache()		
	
	#taking only questions into account by filtering PostTypeId =1
	finalData = finalDF.filter("PostTypeId = 1")
	
	#Attributes that are to be normalized
	cols = ['Score', 'ViewCount', 'AnswerCount', 'CommentCount', 'FavoriteCount']
	
	final = normalizeFeatures(finalDF, cols).cache()
	normalizedData = final.drop("PostId")
	data = normalizedData.drop("PostTypeId")
	
	#set hyper parameters for Kmeans 
	k = 5
	convergeDist = float(1e-2)
	maxNoOfIteration = 100
	WSSSE_list = []
	
	# Create the object of KMeansImplemented class with hyper parameters
	model = KMeansImplemented(k = k, epsilon = convergeDist, maxNoOfIteration = maxNoOfIteration)
	# Calling train method on KMeansImplemented object with entire dataframe
	centers = model.train(data)
	WSSSE = data.map(lambda point: error(point, model)).reduce(lambda x, y: x + y)
	WSSSE_list.append(WSSSE)
	
	# Save the WSSSE 
	plotResults = sc.parallelize(WSSSE_list, 1)
	plotResults.saveAsTextFile("costForEachKImplementedPost")		

	print("Final centers: " + str(kPoints))
	# Predict the cluster index and distance for each userID and store the result
	predictions = final.map(lambda p: (model.clusterAndDistance(np.array(p)[2:], centers), p.UserId))
	predictions.saveAsTextFile("stackOverflowPostOutputImplemented")

	sc.stop()
