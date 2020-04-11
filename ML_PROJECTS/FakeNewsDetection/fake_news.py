#!/usr/bin/env python

import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

filepath = './news.csv'

#Read the data
print("Reading CSV datfile: %s" % (filepath))
df=pd.read_csv(filepath)

## Get shape and head
df.shape
#df.head()

## Get the labels
labels=df.label
lh = labels.head()
#print("Labels: %s\n" % (lh))

# Split the dataset
print("Splitting dataset")
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)


## Initialize a TfidfVectorizer with "Stop" words from the English language and a maximum document 
## frequency of 0.7 (terms with a higher document frequency will be discarded). "Stop" words are 
## the most common words in a language that are to be filtered out before processing the natural 
## language data. And a TfidfVectorizer turns a collection of raw documents into a matrix of TF-IDF features.
print("Initializing a TfidfVectorizer with Stop words...")
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

## Fit and transform the vectorizer on the train set
print("Fit-transforming the vectorizer on the train set...")
tfidf_train=tfidf_vectorizer.fit_transform(x_train)

## Transform the vectorizer on the test set.
print("Transforming the vectorizer on the test set...")
tfidf_test=tfidf_vectorizer.transform(x_test)

## Initialize a PassiveAggressiveClassifier
print("Initializing a PassiveAggressiveClassifier...")
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

## Predict on the test set and calculate accuracy
print("Calculating accuracy of test set predictions...")
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

## Build confusion matrix
print("Building confusion matrix...")
cm = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print("Confusion matrix\n%s" % (cm))
