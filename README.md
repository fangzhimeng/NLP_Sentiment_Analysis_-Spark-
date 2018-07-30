# Sentiment Analysis in Amazon Customer Reviews 
## (Apache Spark on Databricks Community Edition)

###### |<a href='#1'> Introduction </a> | <a href='#2'> Data </a> | <a href='#3'> Preprocessing </a> | <a href='#5'> Model Accuracy </a> |<a href='https://jasonqzhen.github.io/NLP_Sentiment_Analysis_-Spark-/'> Spark Code HTML Version </a>| 

<a id='1'></a>
## Introduction

This project uses the customer review data from Amazon.com to perform a supervised binary (positive or negative) sentiment classification analysis. We use various data pre-processing techniques and three machine learning models, namely, the Naive Bayes model, the Logistic regression model, and the linear support vector classification model.  

The result provides 87% prediction accuracy. 

<a id='2'></a>
## Data

The data comes from the website ["Amazon product data"](http://jmcauley.ucsd.edu/data/amazon/) managed by Dr. Julian McAuley from UCSD. We choose the smaller subset of the customer review data from the Kindle store of Amazon.com [(link to download the dataset)](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Kindle_Store_5.json.gz). The data is in the JSON format, which contains 982,619 reviews and metadata spanning May 1996 - July 2014. 

Reviews with overall rating of 1, 2, or 3 are labeled as negative ("neg"), and reviews with overall rating of 4 or 5 are labeled as positive ("pos"). Thus, number of positive and negative reviews are as follows in the original dataset:

* positive: 829,277 reviews (84.4%)
* negative: 153,342 reviews (15.6%)

We only sample a small portion (8%) of the data for demonstration and try to balance the two classes. After sampling, we have 

* positive: 41,226 reviews (51.7%)
* negative: 38,542 reviews (48.3%)



<a id='3'></a>
## Preprocessing  

The following steps are used to preprocess data:

* Use HTMLParser to un-escape the text
* Change "can't" to "can not", and change "n't" to "not" (This is useful for the later negation handling process)
* Pad punctuations with white spaces
* Lowercase every word 
* Word tokenization
* Word lemmatization
* Perform **negation handling**
   * Use a state variable to store the negation state
   * Transform a word followed by a "not" or "no" into “not_” + word
   * Whenever the negation state variable is set, the words read are treated as “not_” + word
   * The state variable is reset when a punctuation mark is encountered or when there is double negation
* Use bigram and trigram


<a id='5'></a>
## Model accuracy (without parameter tuning)

* Naive Bayes: 85.53%
* Logistic regression: 86.10%
* Linear SVC: 86.56%






