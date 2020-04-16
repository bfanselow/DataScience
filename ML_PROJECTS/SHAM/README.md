# SHAM
## Spam/Ham Analysis Machine
### A simple Spam-Detection exercise in Python3 for learning some of the core concepts of Natural-Language-Processing (NLP) and Document/Text-Classification.  

<img src="https://github.com/bfanselow/DataScience/blob/master/img/spam.jpg" width="100" height="100">

## Project Summary
Two different approaches and code-bases are used for solving the same (Spam-Classification) problem using the same data set:
  * **spam_ham_scratch.py**: Spam-detection using Naive-Bayes algorithm (comparing both TF-IDF and BOW feature-extraction methods). Many of the standard data-science and machine-learning logic blocks are built from scratch as a learning experience.  Some of the code ideas were inspired by an article in: [<u>towardsdatascience.com</u>](https://towardsdatascience.com/spam-classifier-in-python-from-scratch-27a98ddd8e73)
  * **spam_ham_skl.py**: Here we make use of the built-in methods provided by the **sklearn** module for both Naive-Bayes and SVM algorithms. We compare classification results to our homegrown solution.

Several Text-Classification concepts are discussed in the **[Text-Classification.md](https://github.com/bfanselow/DataScience/blob/master/ML_PROJECTS/SHAM/Text-Classification.md)** file.
 
## Data Source 
We use the **[SMS Spam Collection Data Set](https://www.kaggle.com/uciml/sms-spam-collection-dataset)**, a CSV file with each row having a label (spam|ham) and an SMS message. 


## Requires:
 * pip install nltk
 * pip install numpy 
 * pip install pandas 
 * from Python shell download "punkt" tokenizer and english stopwords:
```
   >>> import nltk
       nltk.download(): select "D" for Download
                          ==> select "punkt"
                          ==> select "stopwords"
```

### spam_ham_scratch.py sample run:
```
(venv) $ ./spam_ham_scratch.py
spam_ham_scratch.py (1): Loading data from CSV file: ./spam.csv...
spam_ham_scratch.py (1): Cleaning data in raw dataframe object...
spam_ham_scratch.py (1): Number total messages: 5572. Spam=747, Ham=4825
spam_ham_scratch.py (1): Number training messages: 4223.  Spam=573, Ham=3650
spam_ham_scratch.py (1): Number testing messages: 1349.  Spam=174, Ham=1175
spam_ham_scratch.py (1): Training Naive-Bayes algorithm, feature-extraction method=[BOW]; ngram=2...
spam_ham_scratch.py (1): BOW NB-algorithm training duration (seconds): 19.89
spam_ham_scratch.py (1): Testing NB model, feature-extraction method=[BOW]...
spam_ham_scratch.py (1): BOW NB-model testing duration (seconds): 9.17
=================================
BOW METRICS:
 Total messages tested: 1349
 TP=104  FP=22  TN=1153  FN=70
 Accuracy:  0.93
 Precision:  0.83
 Recall:  0.6
 F1-score:  0.69
Test-Message=[Honey, can you please pick up some dinner on your way home] - Classfication: is-spam=False
Test-Message=[Congratulations you have a chance to win a car. Free entry] - Classfication: is-spam=True
=================================

spam_ham_scratch.py (1): Training Naive-Bayes algorithm, feature-extraction method=[TF-IDF]; ngram=2...
spam_ham_scratch.py (1): TF-IDF NB-algorithm training duration (seconds): 19.88
spam_ham_scratch.py (1): Testing NB model, feature-extraction method=[TF-IDF]...
spam_ham_scratch.py (1): TF-IDF NB-model testing duration (seconds): 9.03
=================================
TF-IDF METRICS:
 Total messages tested: 1349
 TP=125  FP=19  TN=1156  FN=49
 Accuracy:  0.95
 Precision:  0.87
 Recall:  0.72
 F1-score:  0.79
Test-Message=[Honey, can you please pick up some dinner on your way home] - Classfication: is-spam=False
Test-Message=[Congratulations you have a chance to win a car. Free entry] - Classfication: is-spam=True
=================================
```

