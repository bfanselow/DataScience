# SHAM
## Spam/Ham Analysis Machine
### A simple Spam-Detection exercise in Python3 for learning some of the core concepts of Natural-Language-Processing (NLP) and two of the main Text/Document-Classification Machine-Learning models.  

<img src="https://github.com/bfanselow/DataScience/blob/master/img/spam.jpg" width="100" height="100">

## Summary
Two different approaches and code-bases are used for creating and training Spam-Classifier models with the same data set:
  * **spam_ham_scratch.py**: Much of the data-science and machine-learning code is built from scratch as a learninng experience.  Much of the code was inspired by an article in: [<u>towardsdatascience.com</u>](https://towardsdatascience.com/spam-classifier-in-python-from-scratch-27a98ddd8e73)
  * **spam_ham_sklearn.py**: Here we make use of the built-in methods from **sklearn** module, and compare results to our homegrown solution.

Many of the concepts are discussed in **[Text-Classification](https://github.com/bfanselow/DataScience/blob/master/ML_PROJECTS/SHAM/Text-Classification.md)**
 
## Data Source 
We use the **[SMS Spam Collection Data Set](Dataset: https://www.kaggle.com/uciml/sms-spam-collection-dataset)**, a CSV file with each row having a label (spam|ham) and an SMS message. 


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

