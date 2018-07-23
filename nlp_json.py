# Databricks notebook source
import pyspark

# COMMAND ----------

spark.conf.set('spark.sql.shuffle.partitions', '4')

# COMMAND ----------

# load .json file
data_json = spark.sql('''
  SELECT CASE WHEN overall<4 THEN 1
          ELSE 0
          END as class,
        reviewText as text
  FROM kindle_store_5_json
  WHERE length(reviewText)>2''')

# COMMAND ----------

#Sampling
pos = data_json.where('class=0').sample(False, 0.05, seed=1220)
neg = data_json.where('class=1').sample(False, 0.25, seed=1220)
data = pos.union(neg)

# COMMAND ----------

data.groupBy('class').count().show()

# COMMAND ----------

# Define preprocessing function
import html
import string
def clean(row):
    line = html.unescape(row.text)
    line = line.replace("can't", 'can not')
    line = line.replace("n't", " not")
    pad_punct = str.maketrans({key: " {0} ".format(key) for key in string.punctuation}) 
    line = line.translate(pad_punct)  # pad punctuations with white spaces
    line = line.lower()
    line = line.split() # Tokenization
    
    # Negation handling
    # Add "not_" prefix to words behind "not", or "no" until the end of the sentence
    tokens = []
    negated = False
    for t in line:
        if t in ['not', 'no']:
            negated = not negated
        elif t in string.punctuation or not t.isalpha():
            negated = False
        else:
            tokens.append('not_' + t if negated else t)  
    
    return tokens

# COMMAND ----------

# Preprocessing data
from pyspark.sql.functions import col
data_text = data.select('text')
data_class = data.select('class')
data_text_rdd = data_text.rdd.map(clean)
data_text_df = data_text_rdd.zipWithUniqueId().toDF().withColumnRenamed('_1', 'text').withColumnRenamed('_2', 'idx')
data_class_df = data_class.rdd.zipWithUniqueId().toDF().withColumnRenamed('_1', 'class').withColumnRenamed('_2', 'idx')
data_df = data_class_df.join(data_text_df, on='idx')
data_df = data_df.select(col('class').getField('class'), 'text').withColumnRenamed('class.class', 'label')
data_df.show()

# COMMAND ----------

# Generate bigrams and trigrams
from pyspark.ml.feature import NGram
ngram2 = NGram(n=2, inputCol='text', outputCol='bigram')
ngram3 = NGram(n=3, inputCol='text', outputCol='trigram')
data_df = ngram2.transform(data_df)
data_df = ngram3.transform(data_df)
data_df.show()

# COMMAND ----------

# Combine unigram, bigram, and trigram to one column
from itertools import chain
from pyspark.sql.functions import col, udf, size
from pyspark.sql.types import *

def concat(type):
    def concat_(*args):
        return list(chain(*args))
    return udf(concat_, ArrayType(type))

concat_string_arrays = concat(StringType())

data_tokens = data_df.select('label', concat_string_arrays(col("text"), col("bigram"), col("trigram"))).\
  withColumnRenamed('concat_(text, bigram, trigram)', 'tokens').\
  withColumn('length', size(col('tokens')))
data_tokens.show()


# COMMAND ----------

# Split data to 70% for training and 30% for testing
training, testing = data_tokens.randomSplit([0.7,0.3])
training.groupBy('label').count().show()

# COMMAND ----------

# Feature Extraction
from pyspark.ml.feature import CountVectorizer, IDF, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline

count_vec = CountVectorizer(inputCol='tokens', outputCol='c_vec', minDF=5)
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
clean_up = VectorAssembler(inputCols=['tf_idf', 'length'], outputCol='features')
data_prep_pipe = Pipeline(stages=[count_vec, idf, clean_up])

# Define dictionary using training data
cleaner = data_prep_pipe.fit(training)

# Feature Extraction for training data
clean_train = cleaner.transform(training)
clean_train = clean_train.select(['label','features'])
clean_train.show()

# COMMAND ----------

# Feature Extraction for testing data
clean_test = cleaner.transform(testing)
clean_test = clean_test.select(['label','features'])
clean_test.show()

# COMMAND ----------

# Naive Bayes model
from pyspark.ml.classification import NaiveBayes

nb = NaiveBayes()
model_nb = nb.fit(clean_train)
test_nb = model_nb.transform(clean_test)
test_nb.show()

# COMMAND ----------

# Naive Bayes model accuracy
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_eval = MulticlassClassificationEvaluator()
acc_nb = acc_eval.evaluate(test_nb)
print("Accuracy of the model: {}".format(acc_nb))

# COMMAND ----------

# Logistic Regression model
from pyspark.ml.classification import LogisticRegression
lgr = LogisticRegression(maxIter=5)
model_lgr = lgr.fit(clean_train)
test_lgr = model_lgr.transform(clean_test)

# COMMAND ----------

# Logistic Regression model accuracy
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#acc_eval = MulticlassClassificationEvaluator()
acc_lgr = acc_eval.evaluate(test_lgr)
print("Accuracy of the model: {}".format(acc_lgr))

# COMMAND ----------

# Linear SVC model
from pyspark.ml.classification import LinearSVC
lsvc = LinearSVC(maxIter=5)
model_lsvc = lsvc.fit(clean_train)
test_lsvc = model_lsvc.transform(clean_test)

# COMMAND ----------

# Linear SVC model accuracy
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#acc_eval = MulticlassClassificationEvaluator()
acc_lsvc = acc_eval.evaluate(test_lsvc)
print("Accuracy of the model: {}".format(acc_lsvc))

# COMMAND ----------


