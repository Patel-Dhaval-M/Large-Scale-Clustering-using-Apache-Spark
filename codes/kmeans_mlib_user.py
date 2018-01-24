import sys
import numpy as np
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
import xml.etree.ElementTree as ET
from pyspark.sql.functions import stddev_pop, avg, broadcast
from pyspark.sql.functions import *
import re
from pyspark.mllib.clustering import KMeans

def preProcessUsers(data):
	""" method will take single row of user as input in xml format 
	and returns a list with the required attributes"""
	try:
		root = ET.fromstring(data)
	        return [int(root.attrib['Id']), int(root.attrib['Reputation']), int(root.attrib['Views']), int(root.attrib['UpVotes']), int(root.attrib['DownVotes']), int(root.attrib['Age'])]
	except:
		print("Ignoring record")

def preProcessBadges(data):
	""" method will take single row of badges as input in xml format
	and returns a list with the required attributes"""
	try:
		root = ET.fromstring(data)
	        return [int(root.attrib['UserId']), root.attrib['Name']]
	except:
		print("Ignoring record")

def preProcessPosts(data):
	""" method will take single row of posts as input in xml format 
	and returns a list with the required attributes"""
	try:
		root = ET.fromstring(data)
	        return [int(root.attrib['Id']), int(root.attrib['PostTypeId']), root.attrib['OwnerUserId']]
	except:
		print("Ignoring record")

def preProcessComments(data):
	""" method will take single row of comments as input in xml format 
	and returns a list with the required attributes"""
	try:
		root = ET.fromstring(data)
	        return [int(root.attrib['Id']), int(root.attrib['UserId'])]
	except:
		print("Ignoring record")

def oneHotName(data):
	""" method will take an entire rdd of badges and returns 
	the dataframe with userId and one hot encoding of the badges for each userId """
	df = data.toDF(['UserId', 'Name'])	
	distinctColumn = df.select("Name").distinct()
	columns_temp = [str(i.Name) for i in distinctColumn.collect()]	
	columns = [re.sub('[^a-zA-Z0-9+]', '_', x) for x in columns_temp]
	groupedData = data.groupBy(lambda p: p[0])
	processedData = groupedData.map(lambda p :(p[0], helper(p[1], columns)))
	finalDF = processedData.map(lambda (key, data): Row(key, *[eachColumn for eachColumn in data])).toDF(['UserId'] + columns)
	return finalDF

def helper(val, columns):
	""" Helper function used in oneHotName method to set the column as 1 for each badge of the user"""
	a = [0] * len(columns)
	for i in val:
		match = re.sub('[^a-zA-Z0-9+]', '_', i[1])
		a[columns.index(match)] = 1
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
	sc = SparkContext(appName="kmeans_mlib")
	sqlContext = SQLContext(sc)

	# Initially we were trying to use databricks xml package but it was not working because of java heap size memory error
	#users = sqlContext.read.format("com.databricks.spark.xml").option("rowTag","users").load("/data/stackoverflow/Users").selectExpr("explode(row) as users")	
	users = sc.textFile("/data/stackoverflow/Users", 50)
	badges = sc.textFile("/data/stackoverflow/Badges", 50)
	posts = sc.textFile("/data/stackoverflow/Posts", 50)
	comments = sc.textFile("/data/stackoverflow/Comments", 50)	

	usersDF = users.map(preProcessUsers).filter(lambda x: x is not None).toDF(['UserId', 'Reputation', 'Views', 'UpVotes', 'DownVotes', 'Age'])
	
	postsData = posts.map(preProcessPosts).filter(lambda x: x is not None).toDF(['Id', 'PostTypeId', 'UserId'])
	# Filter the answers from all the posts(remove the question)
	filteredPostData = postsData.filter("PostTypeId = 2")
	# Count the number of answers given by each user
	postsDF = filteredPostData.groupBy("UserId").count().withColumnRenamed("count", "post_count")

	commentsData = comments.map(preProcessComments).filter(lambda x: x is not None).toDF(['Id', 'UserId'])
	# Count the number of comments given by each user
	commentsDF = commentsData.groupBy("UserId").count().withColumnRenamed("count", "comment_count")

	badgesData = badges.map(preProcessBadges).filter(lambda x: x is not None)
	badgesDF = oneHotName(badgesData)

	# Left outer join all the dataset with the userId, Use fllna(0) to fill all the null values with zero
	finalData = usersDF.join(postsDF, ["UserId"], "left_outer").join(commentsDF, ["UserId"], "left_outer").join(badgesDF, ["UserId"], "left_outer").fillna(0)
	
	# Normalize all the columns except UserId and one hot encoded features
	cols = ["Reputation", "Views", "UpVotes", "DownVotes", "Age", "post_count", "comment_count"]
	# Cache the result beacuse it avoid preprocessing after  first iteration of kmeans
	normalizedData = normalizeFeatures(finalData, cols).cache()
	
	# remove userid from the DF 
	data = normalizedData.drop(normalizedData.UserId)
	
	# convert dataframe to rdd for mlib input
	datardd1 = data.rdd.map(lambda line: tuple([int(x) for x in line]))

	#set hyper parameters for Kmeans 	
	k = 5
	maxNoOfIteration = 100	
	WSSSE_list = []

	clusters = KMeans.train(datardd1, k, maxIterations=maxNoOfIteration, initializationMode="random")
	# Save WSSSE 
	WSSSE = datardd1.map(lambda point: error(point)).reduce(lambda x, y: x + y)
	WSSSE_list.append(WSSSE)
	plotResults = sc.parallelize(WSSSE_list, 1)
	plotResults.saveAsTextFile("costForEachKMlib")

	# Save result with userID and prediction of cluster index
	tempdatardd = normalizedData.rdd.map(lambda line: (int(line[0]), tuple([int(x) for x in line[1:]])))
	datardd2 = tempdatardd.map(lambda (key, val): (clusters.predict(val), key))	
	datardd2.saveAsTextFile("stackOverflowUserOutputMlib")
	
	sc.stop()
