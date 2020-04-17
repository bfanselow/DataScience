#!/usr/bin/env python
"""

  File: spam_ham_skl.py
  Description:
    Spam/Ham classfication using much of the Machine-Learning logic (algorithms, metrics, etc.)
    built into the Sklearn module. 
    Compariing this to the spam_ham_skl.py script demonstrates the enourmous value provide by
    the sklearn module (both in coding workload reduction and performance impovement) 
  
  Script inspired by: https://www.kaggle.com/pablovargas/naive-bayes-svm-spam-filtering/comments
  
  1) Naive-Bayes-Classifier (NBC) 
    * Train different bayes models changing the regularization parameter α.
    * Evaluate the accuracy, recall and precision of the model with the test set.
    * Select model with best accuracy from thosse models with the highest precision
  
  2) Support Vector Machine (SVM)
    *  Apply the same reasoning applying the support vector machine model with the NBC gaussian kernel.
    *  Train different models changing the regularization parameter C.
    *  Evaluate the accuracy, recall and precision of the model with the test set.
    *  Select the model with highest accuracy from those with the highest precision
  
  The best model is SVM with 98.3% accuracy.
     100% non-spam message correctly classified (precision)
     87.7% of spam messages correctly (recall)

"""
import sys
import os 

import numpy as np
import pandas as pd
import argparse
from collections import Counter
from sklearn import feature_extraction, model_selection, naive_bayes, metrics, svm

import warnings
warnings.filterwarnings("ignore")

myname = os.path.basename(__file__)

DEBUG = 0

##-----------------------------------------------------------------------------------
def dprint(level, msg):
  """
   Print <msg> if level <= DEBUG
  """
  if level <= DEBUG:
    print("%s (%d): %s" % (myname, level, msg))

##-----------------------------------------------------------------------------------
def dprint_object(level, title, object):
  """
   Print object with title if level <= DEBUG
  """
  if level <= DEBUG:
    print("==========================================================================")
    print("%s:" % title)
    print(object)
    print("\n")

##-----------------------------------------------------------------------------------
def load_csv_data_to_dataframe(csv_file):
  """
   Load data from passed csv file into pd dataframe.
   Required args: path to csv_file
   Return dataframe object
  """
  dprint(1, "Loading data from CSV file to dataframe object: %s..." % (csv_file))

  if not os.path.exists(csv_file):
    print("%s: ERROR - Data CSV file not found: [%s]" % (myname, csv_file))
    sys.exit(1)
  if not os.access(csv_file, os.R_OK):
    print("%s: ERROR - Data CSV file cound not be read: [%s]" % (myname, csv_file))
    sys.exit(1)
  df = pd.read_csv(csv_file, encoding='latin-1')

  return( df )

##-----------------------------------------------------------------------------------
def process_data(df):
  """
   Peform several data processing steps including "vetorization", stopword filtering
   and then split into training dataset and testing dataset
   Required Args (pd.dataframe): raw message data 
   Return (tuple): ( X_train, X_test, y_train, y_test) 
  """

  dprint(2, "Processing message data...")

  f = feature_extraction.text.CountVectorizer(stop_words='english')
  csr_matrix = f.fit_transform(df_messages["v2"])
  ## The new feature j in the row i = 1 if the word w_j appears in the text example i or zero if not.
  #print(np.shape(csr_matrix))

  ## Transform the variable spam/non-spam into binary variable
  df_messages["v1"] = df_messages["v1"].map({'spam':1,'ham':0})
  ## Split our data set intio training set and test set.
  X_train, X_test, y_train, y_test = model_selection.train_test_split(csr_matrix, df_messages['v1'], test_size=0.33, random_state=42)
  #print([np.shape(X_train), np.shape(X_test)])

  return( X_train, X_test, y_train, y_test) 

##-----------------------------------------------------------------------------------
def parse_args():
  """ Parse input args are return argparse.parser """
  parser = argparse.ArgumentParser( prog=myname,
                                    usage='%(prog)s <csv-file-path> [options]',
                                    description='Deomnstrate a homegrown NBC spam filter'
                                  )
  parser.add_argument(
                        'Path',
                        metavar='path',
                        type=str,
                        help='Path to CSV file'
                     )
  parser.add_argument(
                        '-d', '--debug',
                        required=False,
                        metavar='debug',
                        type=int,
                        help='Set debug level'
                     )
  parser.add_argument(
                        '-e', '--explore',
                        action='store_true',
                        required=False,
                        help='Explore the dataset'
                      )
  pargs = parser.parse_args()

  return(pargs)

##-----------------------------------------------------------------------------------
if __name__ == '__main__':

  CSV_FILE = None
  MODE = 'model'

  pargs = parse_args()
  dprint(2, "INPUT-ARGS: %s" % (vars(pargs)))

  csv_file = pargs.Path

  if pargs.debug:
    DEBUG = pargs.debug
  if pargs.explore:
    MODE = 'explore'

  ## Load data
  df_messages = load_csv_data_to_dataframe(csv_file)
  #print( df_messages.head() )

  X_train, X_test, y_train, y_test = process_data(df_messages)

  ## Naive-Bayes-Classifier 
  ## We train different bayes models changing the regularization parameter α.
  ## We evaluate the accuracy, recall and precision of the model with the test set.
  list_alpha = np.arange(1/100000, 20, 0.11)
  score_train = np.zeros(len(list_alpha))
  score_test = np.zeros(len(list_alpha))
  recall_test = np.zeros(len(list_alpha))
  precision_test= np.zeros(len(list_alpha))
  count = 0
  for alpha in list_alpha:
    bayes = naive_bayes.MultinomialNB(alpha=alpha)
    bayes.fit(X_train, y_train)
    score_train[count] = bayes.score(X_train, y_train)
    score_test[count]= bayes.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, bayes.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, bayes.predict(X_test))
    count = count + 1  

  ## Check metrics of the first 10 learning models
  matrix = np.matrix(np.c_[list_alpha, score_train, score_test, recall_test, precision_test])
  models = pd.DataFrame(data = matrix, columns = ['alpha', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
  m = models.head(n=10)
  dprint_object(0, "NBC models with changing ALPHA", m)

  ## Select the model with the best test precision
  best_index = models['Test Precision'].idxmax()
  models.iloc[best_index, :]

  ## See if there is more than one model with 100% precision
  df = models[models['Test Precision']==1].head(n=5)
  dprint_object(0, "NBC models with Precision=1", df)

  ## Select model with best accuracy from thosse models with the highest precision
  best_index = models[models['Test Precision']==1]['Test Accuracy'].idxmax()
  bayes = naive_bayes.MultinomialNB(alpha=list_alpha[best_index])
  bayes.fit(X_train, y_train)
  df = models.iloc[best_index, :]
  dprint_object(0, "NBC Highest Accuracy among with highest precision", df)

  ## Confusion matrix with NBC 
  m_confusion_test = metrics.confusion_matrix(y_test, bayes.predict(X_test))
  df = pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])
  dprint_object(0, "NBC Confustion Matrix", df)

  ##-------
  ## Support Vector Machine (SVM)
  ##  Apply the same reasoning applying the support vector machine model with the gaussian kernel.
  ##  Train different models changing the regularization parameter C.
  ##  Evaluate the accuracy, recall and precision of the model with the test set.
  list_C = np.arange(500, 2000, 100) #100000
  score_train = np.zeros(len(list_C))
  score_test = np.zeros(len(list_C))
  recall_test = np.zeros(len(list_C))
  precision_test= np.zeros(len(list_C))
  count = 0
  for C in list_C:
    svc = svm.SVC(C=C)
    svc.fit(X_train, y_train)
    score_train[count] = svc.score(X_train, y_train)
    score_test[count]= svc.score(X_test, y_test)
    recall_test[count] = metrics.recall_score(y_test, svc.predict(X_test))
    precision_test[count] = metrics.precision_score(y_test, svc.predict(X_test))
    count = count + 1 

  ## Check metrics for first 10 learning models
  matrix = np.matrix(np.c_[list_C, score_train, score_test, recall_test, precision_test])
  models = pd.DataFrame(data = matrix, columns = ['C', 'Train Accuracy', 'Test Accuracy', 'Test Recall', 'Test Precision'])
  m = models.head(n=10)
  dprint_object(0, "SVM models", m)

  ## Select the model with the best test precision
  ## The best model zero false positive
  best_index = models['Test Precision'].idxmax()
  df = models.iloc[best_index, :]
  dprint_object(0, "SVM Max Precision", df)

  ## Is there more than one model with 100% precision?
  df = models[models['Test Precision']==1].head(n=5)
  dprint_object(0, "SVM models with Precision=1", df)

  ## Select the model with highest accuracy from those with the highest precision
  best_index = models[models['Test Precision']==1]['Test Accuracy'].idxmax()
  svc = svm.SVC(C=list_C[best_index])
  svc.fit(X_train, y_train)
  df = models.iloc[best_index, :]
  dprint_object(0, "SVM Highest Accuracy among with highest precision", df)

  ## Confusion matrix with SVM classifier
  m_confusion_test = metrics.confusion_matrix(y_test, svc.predict(X_test))
  df = pd.DataFrame(data = m_confusion_test, columns = ['Predicted 0', 'Predicted 1'], index = ['Actual 0', 'Actual 1'])
  dprint_object(0, "SVM Confustion Matrix", df)
  
