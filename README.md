# Large Scale Clustering on Stack overflow dataset using Apache Spark

This project aims at applying K-means clustering algorithm on the StackOverflow dataset to group similar Users and Posts. Appropriate features are selected to extract skill set of Users and relevance of the Posts. 

Algorithm is completely implemented on PySpark to make use of parallel computation of spark and HDFS. Code is implemented without using Mlib library of Spark, results are discussed and finally it is compared with the results obtained after using Sparks Machine learning Mlib library. 

Elbow Method is applied to obtain the optimal number of Cluster for both user and posts dataset. Additionally, two other functions are written to normalize the data and to implement One Hot notations for String Type Data (eg. Badges, Tags)

Project was created as a coursework of Big Data Processing module at Queen Mary university of London. Please check the report Large scale clustering for further details.
