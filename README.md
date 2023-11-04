# Text Classification using Apache Spark

The goal of that project was the implementation of a multi-layer perceptron classifier, using Spark and HDFS, in order, a huge collection of complaints about consumer financial products and services, to be categorized [(Customer Complaint Database)](https://catalog.data.gov/dataset/consumer-complaint-database).

## File Format

The dataset includes complains from customers from 2011 untill today and it it is a comma-delimited .csv file with the following format:

```
0 <- date %Y-%m-%d
1 <- category
2 <- comment
```

## Usage

Given that Spark and HDFS are properly installed and working on our system:

- Upload data file in HDFS
```
hadoop fs -put ./customer_complaints.csv hdfs://master:9000/customer_complaints.csv
```

- Submit job in a Spark environment
```
spark-submit ml.py
```

## Libraries used

nltk, pyspark
