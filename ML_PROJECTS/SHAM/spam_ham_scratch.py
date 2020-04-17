#!/usr/bin/env python
"""

  File: spam_ham_scratch.py
  Description:
   Using our SpamHamClassifier class, train a Naive-Bayes algorithm with
   a large (sub)set of labeled spam|ham messages. Then, test the model using two
   different feature-extraction methods (BOW and TF-IDF). Generate and show
   status for the different train/test runs. Finally, use the model to classify
   a couple test messages


"""
from collections import Counter
import pandas as pd
import numpy as np
import time
import argparse
import sys 
import os

from NB_SpamHamClassifier import SpamHamClassifier
import metrics

myname = os.path.basename(__file__)

DEBUG = 1 

NGRAM = 2 ## N-grams: experiment with changing this from 1-2-3

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
  """
  dprint(1, "Loading data from CSV file: %s..." % (csv_file))

  if not os.path.exists(csv_file):
    print("%s: ERROR - Data CSV file not found: [%s]" % (myname, csv_file))
    sys.exit(1)
  if not os.access(csv_file, os.R_OK):
    print("%s: ERROR - Data CSV file cound not be read: [%s]" % (myname, csv_file))
    sys.exit(1)
  df = pd.read_csv(csv_file, encoding='latin-1')

  return( df )

##-----------------------------------------------------------------------------------
def clean_dataframe(df):
  """
   Load raw dataframe object and clean the data:
     * remove last 3 columns: ('Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4')
     * rename first two columns: ('v1'=>'labels', 'v2'=>'message')
     * re-map labels (spam=>1, ham=>0) by adding first adding binary-value column and then dropping text-label column
   Required args (pd.DataFrame): raw dataframe object. 
   Return (pd.DataFrame): cleaned dataframe object.
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
   Separate the full data-set of messages into "training" and "testing" - Approx 75%/25%. 
   We perform the separation using numpy.random.rand() to create an index-mask on input the dataframe rows:
    * np.random.rand(N) creates an array of shape N and populates it with random samples from a uniform distribution over [0, 1).
    * mask = (np.random.rand(N) < .75) is an array of boolean values identifying if value is above/below .75
    * Therefore: 
        df[mask] will be a dataframe consisting of 75% of the df rows chosen at random 
        df[~mask] will be a dataframe consisting of the other df rows 

   Required args (pd.DataFrame): cleaned dataframe of labeled samples
   Returns (dict): "train" and "test" dataframes
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
def train_test_evaluate(method, df_train_data, df_test_data):
  """
   * Train algorithm with "training" data to create a MODEL
   * Test the model with "testing" data.
   * Evaluate model metrics

   Required args:
    1) method (str): Feature-extraction method 
    2) df_train_data (pd.DataFrame): training data 
    3) df_test_data (pd.DataFrame): testing data 
 
   Return (tuple): SpamHamClassifier Model and dict of test metrics
 
  """
  SHC = SpamHamClassifier(df_train_data, method, NGRAM)

  ## training
  dprint(1, "Training Naive-Bayes algorithm, feature-extraction method=[%s]; ngram=%s..." % (method, NGRAM))
  t_start = time.perf_counter()
  SHC.create_model()
  t_end = time.perf_counter()
  elapsed = t_end - t_start
  dprint(1, "%s NB-algorithm training duration (seconds): %s" % (method, round(elapsed,2)))

  ## testing
  dprint(1, "Testing NB model, feature-extraction method=[%s]..." % (method))
  t_start = time.perf_counter()
  l_predicted_values = SHC.test_model(df_test_data)
  t_end = time.perf_counter()
  elapsed = t_end - t_start
  dprint(1, "%s NB-model testing duration (seconds): %s" % (method, round(elapsed,2)))

  l_known_values = df_test_data['label']
  d_metrics = metrics.gen_model_metrics(l_known_values, l_predicted_values)

  return( (SHC,d_metrics) )

##-----------------------------------------------------------------------------------
def explore_data(df_messages):
  """ 
   Show the TOP-<N> most common spam and ham words from all messages.
   TODO: coordinate with the same logic used by training/testing code with respect to
         which words are filtered, etc. 
  """ 

  top_N = 20
  min_word_length = 3

  ## Get pd.Series for each message type (spam|ham)
  sr_ham_messages = df_messages[df_messages.label==0]["message"]
  sr_spam_messages = df_messages[df_messages.label==1]["message"]

  d_categories = {
    'Spam': sr_spam_messages,
    'Ham': sr_ham_messages
  }

  for cat,pd_sr in d_categories.items():
    column_head = cat + '-words'
    all_words = [] 
    for message in pd_sr:
      words = message.lower().split()
      all_words += words 

    d_words = Counter(all_words)
    l_keys = d_words.keys()
    l_remove = [] 
    for k in l_keys:
      if k.isalpha() == False:
        l_remove.append(k) 
      elif len(k) < min_word_length:
        l_remove.append(k) 

    # pop() + list comprehension to remove multiple keys from the dictionary 
    [d_words.pop(key) for key in l_remove] 

    d_words = d_words.most_common(top_N) 
    df = pd.DataFrame.from_dict(d_words)
    df = df.rename(columns={0: column_head, 1 : "count"})
    print("\nTOP-%d %s" % (top_N, column_head))
    df.index = np.arange(1, len(df) + 1) ## re-index with 1 as start
    print(df)

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
    
  l_test_messages = [  
      'Honey, can you please pick up some dinner on your way home',  ## HAM
      'Congratulations you have a chance to win a car. Free entry'   ## SPAM
  ]
 
  pargs = parse_args() 
  dprint(2, "INPUT-ARGS: %s" % (vars(pargs)))

  csv_file = pargs.Path

  if pargs.debug:
    DEBUG = pargs.debug
  if pargs.explore:
    MODE = 'explore' 

  ## Load data 
  dprint(2, "Loading data...")
  df_messages = load_csv_data_to_dataframe(csv_file) 
  ##print( df_messages.head() )

  ## Clean data
  df_messages = clean_dataframe(df_messages) 
  #print( df_messages.head() )
  
  total_msgs = len(df_messages) ## i.e. df_messages.shape[0]
  total_spam = df_messages[df_messages.label == 1].shape[0]
  total_ham = df_messages[df_messages.label == 0].shape[0]
  dprint(1, "Number total messages: %d. Spam=%d, Ham=%d" % (total_msgs, total_spam, total_ham))

  #print(df_messages['label'].value_counts())

  ## Separate messages into "training" and "testing" sets
  d_separated = separate_training_testing_data(df_messages)

  ## Pandas dataframes
  df_train = d_separated['train']
  df_test = d_separated['test']
  
  N_spam_train = df_train[df_train.label == 1].shape[0]
  N_ham_train = df_train[df_train.label == 0].shape[0]
  dprint(1, "Number training messages: %d.  Spam=%d, Ham=%d" % (len(df_train), N_spam_train, N_ham_train ))

  N_spam_test = df_test[df_test.label == 1].shape[0]
  N_ham_test = df_test[df_test.label == 0].shape[0]
  dprint(1, "Number testing messages: %d.  Spam=%d, Ham=%d" % (len(df_test), N_spam_test, N_ham_test ))
  
  if MODE == 'explore': ## explore the data in various ways
    explore_data(df_messages) 

  else: ## MODE = 'model' - train algorthm, test the model and show performance metrics
    for fa_method in [ 'BOW', 'TF-IDF']:
      model, d_metrics = train_test_evaluate(fa_method, df_train, df_test)
      print("\n=================================")
      print("%s METRICS:" % (fa_method))
      print(" Total messages tested: %d" %  (d_metrics['total']))
      print(" TP=%d  FP=%d  TN=%d  FN=%d" %  (d_metrics['TP'], d_metrics['FP'],d_metrics['TN'],d_metrics['FN']))
      print(" Accuracy: ",  round(d_metrics['accuracy'],2))
      print(" Precision: ", round(d_metrics['precision'],2))
      print(" Recall: ",  round(d_metrics['recall'],2))
      print(" F1-score: ",  round(d_metrics['f1score'],2))
 
      for msg in l_test_messages: 
        result = model.classify_message(msg)
        print("Test-Message=[%s] - Classfication: is-spam=%s" % (msg, result))
      print("\n=================================\n")
