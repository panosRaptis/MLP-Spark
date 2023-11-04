# imports
from pyspark.sql import SparkSession
import datetime as dt
import pandas as pd
import math
from pyspark.ml.linalg import SparseVector
import re
from nltk.corpus import stopwords

from pyspark.ml.feature import StringIndexer
from pyspark.mllib.stat import Statistics
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


def calculate_TF_IDF(f_t_d, max_f, N, n_docs):
	tf = 0.5 + (0.5*f_t_d/max_f)
	idf = math.log(N/n_docs, 2)
	return round(tf*idf, 5)

def my_sort(x):
	list1, list2 = x[0], x[1]
	list1, list2 = zip(*sorted(zip(list1, list2)))
	return (list1, list2)



voc_size = 60

# init spark session
spark = SparkSession.builder.appName("ml").getOrCreate()

# init spark context
sc = spark.sparkContext

# question 2 rdd
complaints = sc.textFile("hdfs://master:9000/customer_complaints.csv").\
                                filter(lambda x: x.startswith("201", 0, 3)).\
                                map(lambda x: x.split(",")).\
                                filter(lambda x: len(x)>= 3 and x[2] != "").\
                                map(lambda x: (x[1], x[2].lower()))


# broadcast stopwords
broad_stopwords = sc.broadcast(set(stopwords.words("english")))

# question 3 rdd remove  
complaints = complaints.map(lambda x: (x[0], re.findall("[a-zA-Z]+", x[1]))).\
                                map(lambda x: (x[0], [word for word in 
                                	x[1] if word not in broad_stopwords.value])).cache()
N = complaints.count()


voc = complaints.flatMap(lambda x: [(y, 1) for y in x[1]]).\
                reduceByKey(lambda x, y: x + y).\
                map(lambda x: (x[1], x[0])).\
               	sortByKey(False).\
               	map(lambda x: x[1]).\
				take(voc_size)

broad_voc = sc.broadcast(voc)

complaints = complaints.map(lambda x: (x[0], [broad_voc.value.index(y) for y in x[1] if y in broad_voc.value])).\
							filter(lambda x: len(x[1]) > 0).\
                        	zipWithIndex().\
							flatMap(lambda x: [((y, x[0][0], x[1]), 1) for y in x[0][1]]).\
							reduceByKey(lambda x, y: x + y).\
							map(lambda x: (x[0][0], [(x[0][2], x[0][1], x[1])])).\
							reduceByKey(lambda x, y: x + y).\
							flatMap(lambda x: [((x[1][i][0], x[1][i][1]), [(x[0], x[1][i][2], len(x[1]))]) 
								for i in range(len(x[1]))]).\
							reduceByKey(lambda x, y: x+y).\
							map(lambda x: (x[0][1], list(map(list, zip(*x[1]))))).\
							map(lambda x: (x[0], (x[1][0], 
								[calculate_TF_IDF(x[1][1][i], max(x[1][1]), N, x[1][2][i])
									for i in range(len(x[1][0]))]))).\
							map(lambda x: (x[0], my_sort(x[1]))).\
							map(lambda x: (x[0], SparseVector(voc_size, x[1][0], x[1][1])))



# covert the final RDD to Spark DataFrame
data = complaints.toDF(["Complaint_Label", "Features"])

# Merge Categories based on the following dictionery
my_dict = {"Credit reporting credit repair services or other personal consumer reports": "Credit reporting repair or other",
            "Credit reporting": "Credit reporting repair or other",
            "Credit card": "Credit card or prepaid card",
            "Prepaid card": "Credit card or prepaid card",
            "Payday loan": "Payday loan title loan or personal loan",
            "Money transfers": "Money transfer virtual currency or money service",
            "Virtual currency": "Money transfer virtual currency or money service"}

data = data.replace(my_dict, 1, "Complaint_Label")

stringIndexer = StringIndexer(inputCol = "Complaint_Label", outputCol = "label")
stringIndexer.setHandleInvalid("skip")
stringIndexerModel = stringIndexer.fit(data)
data = stringIndexerModel.transform(data)

# stratify split
unique_labels = data.select("label").distinct().collect()
dict_label = {}
for i in unique_labels:
    dict_label[i[0]] = 0.7

train_set = data.sampleBy("label", fractions = dict_label, seed = 10)
train_set.cache()
test_set = data.subtract(train_set)

# Multi-Layer Perceptron (MLP) Classifier
layers = [voc_size, 50, 40, 30, len(dict_label)]
trainer = MultilayerPerceptronClassifier(maxIter = 100, layers = layers, blockSize = 128, seed = 1234,
	featuresCol = "Features", labelCol = "label")

# fitting model
model = trainer.fit(train_set)
result = model.transform(test_set)
predictionAndLabels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName = "accuracy")

# printing the results
print("-----------------------  Results  -----------------------")
print("		Statistics\n\n")
print("> Test Set Accuracy = ", 100*float(evaluator.evaluate(predictionAndLabels)), "%")
print("> Total Number of Samples in Train Set = ", train_set.count())
print("> Total Number of Samples in Test Set = ", test_set.count(), "\n\n\n")
print("		More Infos about Stratified Spliting\n\n")
print("> Table with Number of Samples per Class on Train Set:")
train_set.groupBy("Complaint_Label").count().withColumnRenamed("Complaint_Label", "Complaint Label (Train Set)").\
withColumnRenamed("count", "#samples").show(20, False)
print("\n\n> Table with Number of Samples per Class on Test Set:")
test_set.groupBy("Complaint_Label").count().withColumnRenamed("Complaint_Label", "Complaint Label (Test Set)").\
withColumnRenamed("count", "#samples").show(20, False)
