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

