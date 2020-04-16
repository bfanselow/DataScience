"""

 File: NB_SpamHamClassifier.py
 Class: SpamHamClassifier
 Author: Bill Fanselow 2020-03-10

 Description:
  Provides the "SpamHamClassifier" for creating and testing Text-Classification models using 
  a Naive-Bayes algorithm.  We train the Machine-Learning algorithm with data to create a "model".
  Two feature-extraction methods are provide ('Bag-of-Words' and 'TF-IDF').

  -------------------------------------------------------------------------------
  Naive-Bayes-Classfication (NBC) with BOW Summary:

  * TRAINING THE ALGORITHM TO CREATING THE MODEL
    Iterate over the labelled spam/ham messages in the training set. For each word **w** in the set:
     1. Compute P(w|S): (N-spam-messages-containing-w)+1 / (N-spam-messages)+2
     2. Compute P(w|S): (N-ham-messages-containing-w)+1 / (N-ham-messages)+2
     3. Compute P(S):   (N-spam-messsages)/(total-messages)
     4. Compute P(H):   (N-ham-messages)/(total-messages)

  * TESTING THE MODEL
    Iterate over a set of (unlabelled) test messages. For each message:
     1. Create a set {w1,...,wN} of the distinct words in the message.
     2. Compute P(S|w1 ∩ w2 ∩ ...∩ wN) =  P(w1|S)*P(S) * P(w2|S)*P(S)*...*P(wN|S)*P(S) / P(w1)*P(w2)*...*P(wN)
     3. Compute P(H|w1 ∩ w2 ∩...∩ wN) = P(w1|H)*P(H) * P(w2|H)*P(H)*...*P(wN|H)*P(H) / P(w1)*P(w2)*...*P(wN)
     4. Classify as SPAM|HAM by comparing 2 and 3.
        Since we are comparing, we can get rid of the denominator and just see which is numerator is bigger.
        * SPAM if: (P(w1|S)*P(S))*(P(w2|S)*P(S))*...*P(wN|S)*P(S) > (P(w1|H)*P(H))*(P(w2|H)*P(H))*...*P(wN|H)*P(H)
        * HAM  if: (P(w1|S)*P(S))*(P(w2|S)*P(S))*...*P(wN|S)*P(S) > (P(w1|H)*P(H))*(P(w2|H)*P(H))*...*P(wN|H)*P(H)

       To prevent zero probablity results from words never seen in training we *log-transform* the probabilitie:
        * SPAM if: log( (P(w1|S)*P(S))*(P(w2|S)*P(S))*...*P(wN|S)*P(S) ) > log( (P(w1|H)*P(H))*(P(w2|H)*P(H))*...*P(wN|H)*P(H) )
        * HAM  if: log( (P(w1|S)*P(S))*(P(w2|S)*P(S))*...*P(wN|S)*P(S) ) < log( (P(w1|H)*P(H))*(P(w2|H)*P(H))*...*P(wN|H)*P(H) )

       Since log(ab) = log(a) + log(b), our classfication determination becomes:
        * SPAM if: log(P(w1|S)*P(S)) + log((P(w2|S)*P(S))+...+log(P(wN|S)*P(S)) > log(P(w1|H)*P(H)) + log(P(w2|H)*P(H))+...+log(P(wN|H)*P(H))
        * HAM  if: log(P(w1|S)*P(S)) + log((P(w2|S)*P(S))+...+log(P(wN|S)*P(S)) < log(P(w1|H)*P(H)) + log(P(w2|H)*P(H))+...+log(P(wN|H)*P(H))

  -------------------------------------------------------------------------------
  With **TF-IDF**, we are not directly transforming the probabilities of each word. Instead, we can think 
  about it as transforming the documents. With BOW each word in each document counted as 1, whereas with 
  TF-IDF the words in the documents are counted as their TF-IDF weight. We get the known probabilities 
  for Naive Bayes by adding up the TF-IDF weights instead of simply counting the number of words.
  -------------------------------------------------------------------------------

"""
import pandas as pd 
from math import log

import nlp_feature_engineering

FA_METHODS = ['TF-IDF', 'BOW']

##-------------------------------------------------------------------------------------
class InitError(Exception):
    pass

class ModelError(Exception):
    pass

##-------------------------------------------------------------------------------------
class SpamHamClassifier(object):

    def __init__(self, df_train_data, method='TF-IDF', ngram=2):

        if not isinstance(df_train_data, pd.DataFrame):
          raise InitlError("Class must be initialized with pd.dataframe object containing training data")

        if method not in FA_METHODS:
          raise InitError("Class must be initialized with unsupported feature-ext-method: [%s]. Must be one of: %s" (method, str(FA_METHODS)))

        self.messages = df_train_data['message']
        self.labels = df_train_data['label']
       
        self.TRAINED = 0 
        self.method = method 
        self.ngram = ngram 

        ## message counts
        self.spam_messages = self.labels.value_counts()[1]
        self.ham_messages = self.labels.value_counts()[0]
        self.total_messages = self.spam_messages + self.ham_messages

        ## word counts 
        self.count_spam_words = 0
        self.count_ham_words = 0

        self.sum_tf_idf_spam = 0
        self.sum_tf_idf_ham = 0

        ## Conditional probabilities 
        self.prob_word_given_spam = dict()  ## P(w|S) for each word "w"
        self.prob_word_given_ham = dict()   ## P(w|H) for each word "w"

        ## Term frequencies
        self.tf_spam = dict()
        self.tf_ham = dict()

        ## Inverse-document frequencies
        self.idf_spam = dict()
        self.idf_ham = dict()

    #--------------------------------------------------------------------------------    
    def create_model(self):
        """
          Create the Machine-Learning "model" by TRAINING the NBC
        """

        ## Generate term-frequency dicts (tf_spam and tf_ham) and inverse-document-frequency 
        ## dicts (idf_spam and idf_ham)
        self.initialize_features()

        ## Compute probability of Spam: P(S)
        self.prob_spam_message = self.spam_messages / self.total_messages
 
        ## Compute probability of Ham: P(H)
        self.prob_ham_message = self.ham_messages / self.total_messages 

        ## Calculate P(w|S) and P(w|H) for each word "w"
        if self.method == 'TF-IDF':
          self.calc_word_prob_TFIDF()

        elif self.method == 'BOW':
          self.calc_word_prob_BOW()

        self.TRAINED = 1 

    #--------------------------------------------------------------------------------    
    def initialize_features(self):
        """
         As part of algorithm training, iterate over all training messages and populate class attributes:
           * total spam-words and total ham-words  
           * term-frequency dicts (tf_spam and tf_ham)
           * inverse-document-frequency dicts (idf_spam and idf_ham)
        """

        for i in range(self.total_messages):
          word_list = nlp_feature_engineering.message_to_feature_words(self.messages[i], ngram=self.ngram)
          count = list() # keep track of whether the word has ocured in the message or not.
       
          for word in word_list:
            if self.labels[i]: ## SPAM
              self.tf_spam[word] = self.tf_spam.get(word, 0) + 1
              self.count_spam_words += 1
            else: ## HAM
              self.tf_ham[word] = self.tf_ham.get(word, 0) + 1
              self.count_ham_words += 1
            if word not in count:
              count += [word]

          for word in count: 
            if self.labels[i]: ## SPAM
              self.idf_spam[word] = self.idf_spam.get(word, 0) + 1
            else: ## HAM
              self.idf_ham[word] = self.idf_ham.get(word, 0) + 1

    #--------------------------------------------------------------------------------    
    def calc_word_prob_BOW(self):
        """
         As part of algorithm training:
          * Iterate over spam-message term-frequency dict and calculate P(w|S) for each word "w". 
          * Iterate over ham-message term-frequency dict and calculate P(w|H) for each word "w". 
        """

        d_prob_word_given_spam = dict()
        d_prob_word_given_ham = dict()

        ## SPAM: calc P(w|S) for each word "w"
        for word in self.tf_spam:
          d_prob_word_given_spam[word] = (self.tf_spam[word] + 1) / (self.count_spam_words + len(list(self.tf_spam.keys())))
        self.prob_word_given_spam = d_prob_word_given_spam

        ## HAM: calc P(w|H) for each word "w"
        for word in self.tf_ham:
          d_prob_word_given_ham[word] = (self.tf_ham[word] + 1) / (self.count_ham_words + len(list(self.tf_ham.keys())))
        self.prob_word_given_ham = d_prob_word_given_ham

    #--------------------------------------------------------------------------------    
    def calc_word_prob_TFIDF(self):
        """
          As part of algorithm training:
           * Iterate over spam-message term-frequency dict and calculate P(w|S) for each word "w". 
           * Iterate over ham-message term-frequency dict and calculate P(w|H) for each word "w". 
          Store results in class dicts: prob_word_given_spam[word] and prob_word_given_ham[word]
        """

        d_prob_word_given_spam = dict()
        d_prob_word_given_ham = dict()

        sum_tf_idf_spam = 0
        sum_tf_idf_ham = 0

        ## SPAM 
        for word in self.tf_spam:
          d_prob_word_given_spam[word] = (self.tf_spam[word]) * log((self.spam_messages + self.ham_messages) \
                                                          / (self.idf_spam[word] + self.idf_ham.get(word, 0)))
          sum_tf_idf_spam += d_prob_word_given_spam[word]
        for word in self.tf_spam:
          d_prob_word_given_spam[word] = (d_prob_word_given_spam[word] + 1) / (sum_tf_idf_spam + len(list(d_prob_word_given_spam.keys())))
           
        ## HAM 
        for word in self.tf_ham:
          d_prob_word_given_ham[word] = (self.tf_ham[word]) * log((self.spam_messages + self.ham_messages) \
                                                          / (self.idf_spam.get(word, 0) + self.idf_ham[word]))
          sum_tf_idf_ham += d_prob_word_given_ham[word]
        for word in self.tf_ham:
          d_prob_word_given_ham[word] = (d_prob_word_given_ham[word] + 1) / (sum_tf_idf_ham + len(list(d_prob_word_given_ham.keys())))

        self.prob_word_given_spam = d_prob_word_given_spam
        self.prob_word_given_ham = d_prob_word_given_ham
        self.sum_tf_idf_spam = sum_tf_idf_spam 
        self.sum_tf_idf_ham = sum_tf_idf_ham 

    #--------------------------------------------------------------------------------    
    def classify_message(self, message):
        """
          Classify a single message a SPAM/HAM after having trained the algorithm.
          Required arg (str): message text
          Return (boolean): argmax(spam|ham) - True if p_spam >= p_ham, False otherwise

        """
        if not self.TRAINED:
          raise ModelError("Algorithm is not yet trained")
 
        word_list = nlp_feature_engineering.message_to_feature_words(message, ngram=self.ngram)

        p_spam = 0
        p_ham = 0
        for word in word_list:                

          ##
          ## Compute P(S|w1,...wN): 
          ##  P(S|w1,...wN) = log(P(w1|S)*P(S)) + log(P(w2|S)*P(S))+...+log(P(wN|S)*P(S))
          ##                = ( log(P(w1|S)) + log(P(w2|S))+...+log(P(wN|S)) ) + N*log(P(S))
          ##
          if word in self.prob_word_given_spam: ## word is in the training dataset
            p_spam += log(self.prob_word_given_spam[word])

          else: ## word is NOT in the training dataset - apply Laplace smoothing
            ##  P(w_i|S) for each i in the above equation for  P(S|w1,...wN) becomes: ( 1 / count_spam_words + N_w_i)
            ##  Log-transforming this we get: log(1) - log(count_spam_words + N_w_i ) = -log(count_spam_words + N_w_i ) 
            if self.method == 'TF-IDF':
              p_spam -= log(self.sum_tf_idf_spam + len(list(self.prob_word_given_spam.keys())))
            else: ## 'BOW'
              p_spam -= log(self.count_spam_words + len(list(self.prob_word_given_spam.keys())))

          ##
          ## Compute P(H|w1,...wN) 
          ##
          if word in self.prob_word_given_ham: ## word is in the training dataset
            p_ham += log(self.prob_word_given_ham[word])

          else: ## word is NOT in the training dataset
            if self.method == 'TF-IDF':
              p_ham -= log(self.sum_tf_idf_ham + len(list(self.prob_word_given_ham.keys()))) 
            else: ## 'BOW'
              p_ham -= log(self.count_ham_words + len(list(self.prob_word_given_ham.keys())))

          ## This gives us:  N*log(P(S))
          p_spam += log(self.prob_spam_message) 
          p_ham += log(self.prob_ham_message)

        ##print("(%s) P_SPAM=%s   P_HAM=%s" % (message, p_spam, p_ham))

        return( p_spam >= p_ham ) ## True if p_spam >= p_ham, False otherwise

    #--------------------------------------------------------------------------------    
    def test_model(self, df_test_data):
        """
         Pass in a set of (labeled) testing data and pefrom classfication to "test" the model.
         Required Args (pd.dataframe): testing data (i.e. classification predictions)
         Return (dict): classification result (1=spam|0=ham) for each message in df_test_data
        """
        if not self.TRAINED:
          raise ModelError("Algorithm is not yet trained")

        if not isinstance(df_test_data, pd.DataFrame):
          raise InitlError("Input test-data must be contined in pd.dataframe object")
        
        test_messages = df_test_data['message']

        d_test_results = dict()
        for (i, message) in enumerate(test_messages):
          d_test_results[i] = int(self.classify_message(message))

        return( d_test_results )
