# Databricks notebook source
# MAGIC %md # Sentiment Analysis of Amazon Customer Reviews

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC This project uses the customer review data from Amazon.com Kindle store to perform a supervised binary (positive or negative) sentiment classification analysis. We use various data pre-processing techniques and three machine learning models, namely, Naive Bayes classification model, the Logistic regression model, and the linear support vector classification model. The result provides 87% prediction accuracy.

# COMMAND ----------

import pyspark
spark.conf.set('spark.sql.shuffle.partitions', '8')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load dataset
# MAGIC The data comes from the website "Amazon product data" (http://jmcauley.ucsd.edu/data/amazon/) managed by Dr. Julian McAuley from UCSD. We choose the smaller subset of the customer review data from the Kindle store of Amazon.com. The data is in the JSON format, which contains 982,619 reviews and metadata spanning May 1996 - July 2014.

# COMMAND ----------

# load original .json data
kindle_json = spark.read.json('/FileStore/tables/Kindle_Store_5.json')

# COMMAND ----------

display(kindle_json)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate Sentiment Label
# MAGIC 
# MAGIC Reviews with overall rating of 1, 2, or 3 are labeled as negative (label=1), and reviews with overall rating of 4 or 5 are labeled as positive (label=0). 

# COMMAND ----------

kindle_json.createOrReplaceTempView('kindle_json_view')

data_json = spark.sql('''
  SELECT CASE WHEN overall<4 THEN 1
          ELSE 0
          END as label,
        reviewText as text
  FROM kindle_json_view
  WHERE length(reviewText)>2''')

data_json.groupBy('label').count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Generate the dataset for modeling
# MAGIC We only sample a small portion of the data for demonstration and try to balance the two classes.

# COMMAND ----------

# Sampling data
pos = data_json.where('label=0').sample(False, 0.05, seed=1220)
neg = data_json.where('label=1').sample(False, 0.25, seed=1220)
data = pos.union(neg)
data.groupBy('label').count().show()

# COMMAND ----------

# Negative reviews are on average longer than the positive reviews, but not significantly longer
from pyspark.sql.functions import length
data.withColumn('review_length', length('text')).groupBy('label').avg('review_length').show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preprocessing
# MAGIC Data preprocessing process uses the following steps:
# MAGIC 
# MAGIC * Use HTMLParser to un-escape the text
# MAGIC * Change "can't" to "can not", and change "n't" to "not" (This is useful for the negation handling process)
# MAGIC * Pad punctuations with blanks
# MAGIC * Lowercase every word
# MAGIC * Word tokenization
# MAGIC * Word lemmatization
# MAGIC * Perform **negation handling**
# MAGIC     * Use a state variable to store the negation state
# MAGIC     * Transform a word followed by a "not" or "no" into “not_” + word
# MAGIC     * Whenever the negation state variable is set, the words read are treated as “not_” + word
# MAGIC     * The state variable is reset when a punctuation mark is encountered or when there is double negation
# MAGIC * Use **bigram** and/or **trigram** models

# COMMAND ----------

# Define preprocessing function
def clean(text):
    import html
    import string
    import nltk
    nltk.download('wordnet')
    
    line = html.unescape(text)
    line = line.replace("can't", 'can not')
    line = line.replace("n't", " not")
    # Pad punctuations with white spaces
    pad_punct = str.maketrans({key: " {0} ".format(key) for key in string.punctuation}) 
    line = line.translate(pad_punct)
    line = line.lower()
    line = line.split() 
    lemmatizer = nltk.WordNetLemmatizer()
    line = [lemmatizer.lemmatize(t) for t in line] 
    
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
    
    invalidChars = str(string.punctuation.replace("_", ""))  
    bi_tokens = list(nltk.bigrams(line))
    bi_tokens = list(map('_'.join, bi_tokens))
    bi_tokens = [i for i in bi_tokens if all(j not in invalidChars for j in i)]
    tri_tokens = list(nltk.trigrams(line))
    tri_tokens = list(map('_'.join, tri_tokens))
    tri_tokens = [i for i in tri_tokens if all(j not in invalidChars for j in i)]
    tokens = tokens + bi_tokens + tri_tokens      
    
    return tokens

# COMMAND ----------

# An example: how the function clean() pre-processes the input text
example = clean("I don't think this book has any decent information!!! It is full of typos and factual errors that I can't ignore.")
print(example)

# COMMAND ----------

# Perform data preprocessing
from pyspark.sql.functions import udf, col, size
from pyspark.sql.types import ArrayType, StringType
clean_udf = udf(clean, ArrayType(StringType()))
data_tokens = data.withColumn('tokens', clean_udf(col('text')))
data_tokens.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Split dataset to training (70%) and testing (30%) sets

# COMMAND ----------

# Split data to 70% for training and 30% for testing
training, testing = data_tokens.randomSplit([0.7,0.3], seed=1220)
training.groupBy('label').count().show()

# COMMAND ----------

training.cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Naive Bayes Model (with parameter tuning)

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml import Pipeline

count_vec = CountVectorizer(inputCol='tokens', outputCol='c_vec', minDF=5.0)
idf = IDF(inputCol="c_vec", outputCol="features")

# COMMAND ----------

# Naive Bayes model
from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes()

pipeline_nb = Pipeline(stages=[count_vec, idf, nb])

model_nb = pipeline_nb.fit(training)
test_nb = model_nb.transform(testing)
test_nb.show(3)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Naive Bayes model performance (using default parameters)
# MAGIC * Area under the ROC curve: 0.8551
# MAGIC * Accuracy: 0.8553

# COMMAND ----------

# Naive Bayes model ROC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
roc_nb_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='label')
roc_nb = roc_nb_eval.evaluate(test_nb)
print("ROC of the NB model: {}".format(roc_nb))

# COMMAND ----------

# Naive Bayes model accuracy
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_nb_eval = MulticlassClassificationEvaluator(metricName='accuracy')
acc_nb = acc_nb_eval.evaluate(test_nb)
print("Accuracy of the NB model: {}".format(acc_nb))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Naive Bayes model performance after parameter tuning
# MAGIC * CountVectorizer.minDF = 7.0
# MAGIC * NaiveBayes.smooting = 1.0
# MAGIC * Accuracy: 0.8568 (increased from 0.8553)

# COMMAND ----------

# NB parameter tuning and CV
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid_nb = (ParamGridBuilder()
                .addGrid(count_vec.minDF, [3.0, 5.0, 7.0, 10.0, 15.0])
                .addGrid(nb.smoothing, [0.1, 0.5, 1.0])
                .build())
cv_nb = CrossValidator(estimator=pipeline_nb, estimatorParamMaps=paramGrid_nb, evaluator=acc_nb_eval, numFolds=5)
cv_model_nb = cv_nb.fit(training) 

# COMMAND ----------

test_cv_nb = cv_model_nb.transform(testing)
acc_nb_cv = acc_nb_eval.evaluate(test_cv_nb)
print("Accuracy of the NB CV model: {}".format(acc_nb_cv))

# COMMAND ----------

cv_model_nb.bestModel.stages[0].extractParamMap()

# COMMAND ----------

cv_model_nb.bestModel.stages[2].extractParamMap()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regressions
# MAGIC Model performance (using default parameters)
# MAGIC * Area under the ROC curve: 0.8601
# MAGIC * Accuracy: 0.8610

# COMMAND ----------

# Logistic Regression model
from pyspark.ml.classification import LogisticRegression
lgr = LogisticRegression(maxIter=5)
pipeline_lgr = Pipeline(stages=[count_vec, idf, lgr])

model_lgr = pipeline_lgr.fit(training)
test_lgr = model_lgr.transform(testing)

# COMMAND ----------

# Logistic Regression model ROC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
roc_lgr_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='label')
roc_lgr = roc_lgr_eval.evaluate(test_lgr)
print("ROC of the model: {}".format(roc_lgr))

# COMMAND ----------

# Logistic Regression model accuracy
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_lgr_eval = MulticlassClassificationEvaluator(metricName='accuracy')
acc_lgr = acc_lgr_eval.evaluate(test_lgr)
print("Accuracy of the model: {}".format(acc_lgr))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Linear SVC Model
# MAGIC Model performance (using default parameters)
# MAGIC * Area under the ROC curve: 0.8649
# MAGIC * Accuracy: 0.8656

# COMMAND ----------

# Linear SVC model
from pyspark.ml.classification import LinearSVC
lsvc = LinearSVC(maxIter=5)
pipeline_lsvc = Pipeline(stages=[count_vec, idf, lsvc])

model_lsvc = pipeline_lsvc.fit(training)
test_lsvc = model_lsvc.transform(testing)

# COMMAND ----------

# Linear SVC model ROC
from pyspark.ml.evaluation import BinaryClassificationEvaluator
roc_lsvc_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='label')
roc_lsvc = roc_lsvc_eval.evaluate(test_lsvc)
print("ROC of the model: {}".format(roc_lsvc))

# COMMAND ----------

# Linear SVC model accuracy
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator
acc_lsvc_eval = MulticlassClassificationEvaluator(metricName='accuracy')
acc_lsvc = acc_lsvc_eval.evaluate(test_lsvc)
print("Accuracy of the model: {}".format(acc_lsvc))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Predict on new reviews:
# MAGIC To demonstrate the model prediction on new review texts, I randomly choose five reviews from the Kindle book *The Brave Ones: A Memoir of Hope, Pride and Military Service, by Michael J. MacLeod*. 
# MAGIC 
# MAGIC The suffixes "_1", "_2", ..., "_5" indicate the real overall review stars 1, 2, ..., 5.
# MAGIC 
# MAGIC The model correctly predicts the first three reviews as "negative" (label=1), and the last two as "positive" (label=0).

# COMMAND ----------

review_1 = ["WOW!!! No words describe how bland this book is. It took me a lot to even pick up to read. I would definitely not recommend this book."]

# COMMAND ----------

review_2 = ["A first person account of the war in Afghanistan. It skipps around a lot and is like a never-ending news article. On the positive side, you do get a feel for what desert fighting is like from a soldiers point of view."]

# COMMAND ----------

review_3 = ["I liked the premise and most of the book. At the end parts I lost a little interest because I lost the thread of who was who. War is hell. MacLeod did his service unlike most of us."]

# COMMAND ----------

review_4 = ["Very informative first person account of the the daily life of a US Paratrooper. From training to deployment in combat situations in Afghanistan. Well worth the read and makes you really understand and appreciate their sacrifices"]

# COMMAND ----------

review_5 = ["This is perhaps the best wrote book I have ever read. Articulate and thought provoking. Not just a riveting account of actual combat, but Michael was able to do what few before him have...captured the essence of what one feels as the battle unfolds. Perhaps most of all, I am grateful to call this author 'Fellow Warrior' Airborne all the way!!!"]

# COMMAND ----------

from pyspark.sql.types import *
schema = StructType([StructField("text", StringType(), True)])

text = [review_1, review_2, review_3, review_4, review_5]
review_new = spark.createDataFrame(text, schema=schema)

# COMMAND ----------

# Data preprocessing
review_new_tokens = review_new.withColumn('tokens', clean_udf(col('text')))
review_new_tokens.show()

# COMMAND ----------

# Prediction using tuned Naive Bayes model
result = cv_model_nb.transform(review_new_tokens)
result.select('text', 'prediction').show()

# COMMAND ----------


