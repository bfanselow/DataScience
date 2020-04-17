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
---
### Usage
```
(venv) $ ./spam_ham_scratch.py <csv-file-path> [options]

 OPTIONS:
   --explore        (explore the spam|ham dataset)
   -d <debug-level> (change default debug level for additional processing messages)
 
```
### Model performace tuning
Try tweaking pre-processing and feature-engineering parameters in the **nlp_feature_engineering.py** module and restesting model performace. For example, it was found that using n-grams resulted (not surprsingly) in much better performance. However, removing stopwords BEFORE n-gram compilation resulted in **lower** performance!

---
### Example training/testing run:
```
(venv) $ ./spam_ham_scratch.py ./spam.csv -d 2
spam_ham_scratch.py (1): Loading data from CSV file to dataframe object: ./spam.csv...
spam_ham_scratch.py (2): Reformatting raw dataframe object...
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

---
### Example data-exploration run 
```
(venv) $ ./spam_ham_scratch.py ./spam.csv --explore
spam_ham_scratch.py (1): Loading data from CSV file to dataframe object: ./spam.csv...
spam_ham_scratch.py (2): Reformatting raw dataframe object...
spam_ham_scratch.py (1): Number total messages: 5572. Spam=747, Ham=4825
spam_ham_scratch.py (1): Number training messages: 4155.  Spam=556, Ham=3599
spam_ham_scratch.py (1): Number testing messages: 1417.  Spam=191, Ham=1226

TOP-20 Spam-words
   Spam-words  count
1        call    342
2        your    263
3         you    252
4         the    204
5         for    201
6        free    180
7         txt    136
8        have    135
9        from    127
10        and    122
11       text    112
12     mobile    109
13       with    108
14      claim    106
15      reply    101
16        now     93
17       stop     89
18       this     86
19        our     85
20        get     82

TOP-20 Ham-words
   Ham-words  count
1        you   1665
2        the   1113
3        and    845
4        for    496
5       that    442
6       have    433
7       your    413
8        but    413
9        are    405
10       not    381
11       can    356
12      will    331
13       get    293
14      just    286
15      when    270
16      with    269
17       how    245
18      what    235
19       all    231
20       got    227
```
---
```
TOP-20 Spam-words (after stopword removal)
   Spam-words  count
1        call    342
2        free    180
3         txt    136
4        text    112
5      mobile    109
6       claim    106
7       reply    101
8        stop     89
9         get     82
10        new     69
11       send     65
12      nokia     64
13        win     58
14      prize     58
15       cash     56
16    contact     56
17     please     52
18    service     48
19        per     44
20       tone     40
TOP-20 Ham-words (after stopword removal)
   Ham-words  count
1        get    293
2        got    227
3       like    221
4       call    215
5       come    215
6       know    208
7       good    187
8      going    157
9       want    153
10      time    153
11      love    149
12      need    147
13     still    144
14       one    141
15       see    126
16     think    124
17      dont    124
18      send    121
19      tell    119
20      home    110
```
