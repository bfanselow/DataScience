#!/usr/bin/env python
"""


"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import os

from SpamHamClassifier import SpamHamClassifier
import metrics

myname = os.path.basename(__file__)

DEBUG = 1 

##-----------------------------------------------------------------------------------
def dprint(level, msg):
  """
   Print <msg> if level <= DEBUG
  """
  if level <= DEBUG:
    print("%s (%d): %s" % (myname, level, msg))

##-----------------------------------------------------------------------------------
def load_csv_data_to_dataframe(csv_file):
  """
   Load data from passed csv file into pd dataframe.
   Required args: path to csv_file
   Return dataframe object

   TODO: check for file existence and handle errors 
  """
  dprint(1, "Loading data from CSV file: %s..." % (csv_file))
  df = pd.read_csv(csv_file, encoding='latin-1')

  return( df )

##-----------------------------------------------------------------------------------
def clean_dataframe(df):
  """
   Load raw dataframe object and clean the data:
     * remove last 3 columns: ('Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4')
     * rename first two columns: ('v1'=>'labels', 'v2'=>'message')
     * re-map labels (spam=>1, ham=>0) by adding first adding binary-value column and then dropping text-label column
   Required args: raw dataframe object. 
   Return cleaned dataframe object.
  """
  dprint(1, "Cleaning data in raw dataframe object...")

  ## Drop last 3 (unused) cols
  df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

  ## rename fist 2 cols
  df.rename(columns = {'v1': 'labels', 'v2': 'message'}, inplace=True)

  ## remap spam/hame label values
  df['label'] = df['labels'].map({'ham': 0, 'spam': 1})
  df.drop(['labels'], axis = 1, inplace = True)

  return(df)

##-----------------------------------------------------------------------------------
def separate_training_testing_data(df):
  """
   Separate the full data-set of messages into "training" (75%) and "testing" (25%) 
   We perform the separation using numpy.random.rand() to create an index-mask on input the dataframe rows:
    * np.random.rand(N) creates an array of shape N and populates it with random samples from a uniform distribution over [0, 1).
    * mask = (np.random.rand(N) < .75) is an array of boolean values identifying if value is above/below .75
    * Therefore: 
        df[mask] will be a dataframe consisting of 75% of the df rows chosen at random 
        df[~mask] will be a dataframe consisting of the other df rows 

   Required arg: cleaned dataframe of labeled samples
   Returns: dict of "train" and "test" dataframes
  """

  mask = (np.random.rand(len(df)) < .75)

  df_train = df[mask]
  df_test = df[~mask]

  ## reset dataframe row indexes (0...N)
  df_train.reset_index(inplace=True, drop=True)
  df_test.reset_index(inplace=True, drop=True)

  d_data = {
    "train": df_train,
    "test": df_test
  }

  return(d_data)

##-----------------------------------------------------------------------------------
if __name__ == '__main__':

  csv_file = './spam.csv'
 
  ## Load data 
  dprint(2, "Loading data...")
  df_mails = load_csv_data_to_dataframe(csv_file) 
  ##print( df_mails.head() )

  ## Clean data
  df_mails = clean_dataframe(df_mails) 
  #print( df_mails.head() )
  
  total_msgs = len(df_mails) ## i.e. df_mails.shape[0]
  total_spam = df_mails[df_mails.label == 1].shape[0]
  total_ham = df_mails[df_mails.label == 0].shape[0]
  dprint(1, "Total messages: %d. Spam=%d, Ham=%d" % (total_msgs, total_spam, total_ham))

  #print(df_mails['label'].value_counts())

  ## Separate messages into "training" and "testing" sets
  d_separated = separate_training_testing_data(df_mails)

  df_train = d_separated['train']
  df_test = d_separated['test']
  
  N_spam_train = df_train[df_train.label == 1].shape[0]
  N_ham_train = df_train[df_train.label == 0].shape[0]
  dprint(1, "Training messages: %d.  Spam=%d, Ham=%d" % (len(df_train), N_spam_train, N_ham_train ))

  N_spam_test = df_test[df_test.label == 1].shape[0]
  N_ham_test = df_test[df_test.label == 0].shape[0]
  dprint(1, "Testing messages: %d.  Spam=%d, Ham=%d" % (len(df_test), N_spam_test, N_ham_test ))

  ##
  ## Run Classifier with TF-IDF 
  ##
  shc_tfidf = SpamHamClassifier(df_train, 'tf-idf')
  dprint(1, "Training TF-IDF model...")
  t_start = time.perf_counter()
  shc_tfidf.train_model()
  t_end = time.perf_counter()
  elapsed = t_end - t_start
  dprint(1, "TF-IDF model training duration (seconds): %s" % (round(elapsed,2)))

  dprint(1, "Testing TF-IDF model...")
  t_start = time.perf_counter()
  l_predicted_values = shc_tfidf.test_model(df_test['message'])
  t_end = time.perf_counter()
  elapsed = t_end - t_start
  dprint(1, "TF-IDF model testing duration (seconds): %s" % (round(elapsed,2)))

  d_metrics = metrics.gen_model_metrics(df_test['label'], l_predicted_values)

  print("\nTF-IDF METRICS:")
  print("Total messages tested: %d" %  (d_metrics['total']))
  print("TP=%d  FP=%d  TN=%d  FN=%d" %  (d_metrics['TP'], d_metrics['FP'],d_metrics['TN'],d_metrics['FN']))
  print("Accuracy: ",  round(d_metrics['accuracy'],2))
  print("Precision: ", round(d_metrics['precision'],2))
  print("Recall: ",  round(d_metrics['recall'],2))
  print("F1-score: ",  round(d_metrics['f1score'],2))

  ##
  ## Run Classifier with BOW 
  ##
  shc_bow = SpamHamClassifier(df_train, 'bow')
  dprint(1, "Training BOW model...")
  t_start = time.perf_counter()
  shc_bow.train_model()
  t_end = time.perf_counter()
  elapsed = t_end - t_start
  dprint(1, "BOW model training duration (seconds): %s" % (round(elapsed,2)))

  dprint(1, "Testing BOW model...")
  t_start = time.perf_counter()
  l_predicted_values = shc_bow.test_model(df_test['message'])
  t_end = time.perf_counter()
  elapsed = t_end - t_start
  dprint(1, "TF-IDF model testing duration (seconds): %s" % (round(elapsed,2)))

  d_metrics = metrics.gen_model_metrics(df_test['label'], l_predicted_values)

  print("\nBOW METRICS:")
  print("Total messages tested: %d" %  (d_metrics['total']))
  print("TP=%d  FP=%d  TN=%d  FN=%d" %  (d_metrics['TP'], d_metrics['FP'],d_metrics['TN'],d_metrics['FN']))
  print("Accuracy: ",  round(d_metrics['accuracy'], 3))
  print("Precision: ", round(d_metrics['precision'], 3))
  print("Recall: ",  round(d_metrics['recall'], 3))
  print("F1-score: ",  round(d_metrics['f1score'], 3))
