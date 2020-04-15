# Document/Text-Classification Models for Spam Detection 

Given some text input, automatically classify it into predefined catagories (i.e. assign some label to it). This has many important practical uses such as: *spam-filtering, sentiment-analysis, fake-news-detection, genre-classification or comprehension level of a book* and more.

## Background
Machine Learning models are categorized as **Supervised** or **Unsupervised**.  The majority of practical machine learning uses supervised learning, in which you know a set of input data (x1,x2,...x3) and the output variable (Y) and use the algorithm to learn the mapping function from the input to the output:
```
Y = f(x1,x2...xN)
```

The goal is to approximate the mapping function so well that when you have new input data (x) that you can predict the output variables (Y) for that data.

We divide the input data set into two groups "training" data and "testing" data (typically with a 70/30 to 90/10 ratio). The algorithm "learns
" on the training data and then is tested with the testing data. Only when the algorithm can successfully

Supervised learning problems can be further grouped into **regression** and **classification** problems.
 * A *classification* problem is when the output variable is a category, such as ("red" or "blue") or ("spam" and "not-spam").
 * A *regression* problem is when the output variable is a real value, such as "dollars" or "weight".

Unsupervised learning is where you only have input data (x1,x2,...xN) but no corresponding output variables.  The goal for unsupervised learning is to model the underlying structure or distribution in the data in order to learn more about the data.
Unsupervised learning problems can be further grouped into **clustering** and **association** problems.
 * A *clustering* problem is where you want to discover the inherent groupings in the data, such as grouping customers by purchasing behavior.
 * An *association* rule learning problem is where you want to discover rules that describe large portions of your data, such as people that buy X also tend to buy Y.

The problem with interpreting the human language is that it is not a set of rules or binary data that can be fed into a system.
However, with the help of Natural Language Processing (NLP) and different Machine-Learning "models" (classification-algorithms) like Naive Bayes (NB), Support-Vector-Machine (SVM), Boosting-Models and Neural Networks we can implement a set of computational rules.

---
Our Spam/Ham Analysis Machine will use two (Supervised-ML) **Text Classification** models, classifying into two categories: "spam" and "not-spam" (a.k.a. "ham").
 1) **Naive Bayes**
 2) **Support-Vector-Machine**

---
## Model-1: Naive Bayes
First, we consider the basic Bayes Theorem:
```
 P(A|B) = P(A)*P(B|A) / P(B) 
```
This theorem provides way of finding an unknown probability based on certain other probabilities that we do know. "|" means **given that**.
See: https://www.mathsisfun.com/data/bayes-theorem.html

This equation tells us how likely that A will happen **given that** B happened: P(A|B)
When we know the following from previous observation:   
  * how often B happens given that A happens: P(B|A)
  * how likely A is on its own:  P(A)
  * how likely B is on its own: P(B)

For example, if we know:
  1) We have identified by historical observation that dangerous fires are rare on any given day: P(Fire) = 1% 
     (num-days-with-fire-per-year/num-days-per-year)
  2) We have identified that percentage of dangerous fires which produce smoke: P(Smoke|Fire) = 97%
  3) We have identified the probability that smoke will be seen on any given day: P(Smoke) = 9% 
     (num-days-with-smoke-per-year/num-days-per-year)
     (This includes smoke with or without dangerous fire: i.e. barbeques, farmers doing controlled burns, etc.)

What we want to know/predict: 
  How likely is it that there is a dangerous fire when we see some smoke?

Answer:
```
  P(Fire|Smoke) = P(Fire)*P(Smoke|Fire) / P(Smoke) = (.01)*(.97) / .09 = .1077 = 10.8% 
```
  So, based on our historical observations, when we see smoke there is a reasonable chance that we should be concerned about this being due to a dangerous fire.

---
One of the most common uses for Bayes theorem is calculating probability of FALSE-POSITIVES and FALSE-NEGATIVES.

Suppose you take a new test for Protate Cancer. Unfortunately, the test does have a certain number of false-positives and false-negatives.
What we know:
 * 1% of all men in general population are diagnosed prostate cancer. (i.e. 99% of men do not have PC).
 * For men that really do have a tumor, the test says "Yes" 80% of the time
 * For men that do not have a tumor, the test says "Yes" 10% of the time ("false positive")

Question:
  If you take the new PC test and it says "Yes", what are the chances that you really have prostate cancer? 
  We need to calculate the chance of actually having prostate cancer when the PC test says "Yes": P(Cancer|Yes)

Using Bayes Theorem:
```
   P(Cancer|Yes) = P(Cancer)*P(Yes|Cancer)/P(Yes) 
```
Where:
  * P(Cancer) is Probability of a man having prostate cancer = 1%
  * P(Yes|Cancer) is Probability of test saying "Yes" for men with cancer = 80%
  * P(Yes) is Probability of test saying "Yes" (to any man taking the test) = ??%

We have a problem: Since we have not perfomed a random test sampling on a huge number of the general male population, we don't know P(Yes): the probability of the test saying "Yes" for all men.

However, we can calculate it by adding up those with, and those without prostate cancer:
   * 1% of men have PC, and the test says "Yes" to 80% of them.
   * 99% do not have PC and the test says "Yes" to 10% of them.

Therefore:
```
  P(Yes) = (1% * 80%) + (99% * 10%) = 10.7%
```
In other words, about 10.7% of the male population will get a "Yes" result from the test (regardless of PC existence).

So now we can complete our formula:
```
  P(Cancer|Yes) = (1% * 80%) / 10.7% = 7.48% 
```
  
If you take the new PC test and it says "Yes", there is only a 7% chance you actually have prostate cancer. The new test has a 7% FALSE-POSITIVE rate.

What we have done here is to re-write the denominator for a special version of Bayes Theorem when we are unable to measure P(B) directly:
```
 P(A|B) = P(A)*P(B|A) / P(B)  ===>  P(A|B) = P(A)P(B|A) / P(A)P(B|A) + P(!A)P(B|!A)
```

---
In the above case, we had two A-conditions: "A" and "not-A".  Suppose we have 3 or more conditions for A (A1, A2, A3, ...):

Bayes Theorem becomes:
```
  P(A1|B) = P(A1)P(B|A1) /( P(A1)P(B|A1) + P(A2)P(B|A2) + P(A3)P(B|A3) + ...) 
```

As an example consider an art competition which is composed 3 artists (Anne, Alice, and Art):
  * Anne submits 15 paintings - P(Anne) = 15/30 
    4% of Anne's works have won First Prize in previous competitions. In other words, the probability that a painting wins First-place given that the painting is by Anne is 4%
  * Alice submits 5 paintings. 6% of her works have won First Prize in previous competitions.
  * Art submits 10 paintings. 3% of his works have won First Prize in previous competitions.

What is the chance that Anne will win First Prize? Note that there are 30 total paintings.
```
 P(Anne|First) =  P(Anne)P(First|Anne) / P(Anne)P(First|Anne) + P(Alice)P(First|Alice) + P(Art)P(First|Art)
```

We are asking:
 What is probability that the painting is Anne's given that the prize is first-place.

Bayes Equation:
 * Numerator: (prob painting is Anne's)*(prob that the prize is first-place given that painting is Anne's)
 * Denominator: (same-as-numerator  + equivelent-for-Alice + equiv-for-Art) 

```
 P(Anne|First) = (15/30)*4% / (15/30)*4% + (5/30)*6% + (10/30)*3%  = 15*4% / (15*4% + 5*6% + 10*3%) = 0.6/0.6+0.3+0.3 = 50%
```

One ciritcal assumption which allowed us to make these additive calculations in the denominator is that those conditions all have totally independent probabilites. This is true, but is not always the case. When we do make this (naive) assumption, Bayes Classification is referred to as a **Naive Bayes Classification** (NBC).

---

### Applying Bayes Theorem to Spam detection:
In spam filtering/detection we make the assumption that one class of messages (i.e. spam messages in this case) tend to have have a different word distribution than messages in the other class (i.e. non-spam messages). For example, one might imagine the word "free" is more common in spam messages than non-spam. 
```
P(S|w) = P(w|S)*P(S) / P(w)
```
 where:
  * P(S|w):  Probability that a messages is spam, knowing that we have some word "w" in it
  * P(w|S):  Probability of finding that word "w" is in a spam message
  * P(S):    Probability that a message is spam 
  * P(w):    Probability that word "w" is in any message 

This works when we can measure all values, P(w|S),P(S),P(w), directly.

If we are unable to measure P(w) directly, we can calculate this we can use Naive Bayes Classification, since we have two independent conditions - spam or NOT-spam (HAM).
```
P(S|w) = P(w|S)*P(S) / P(w|S)*P(S) + P(w|H)*P(H) 
```
 
If we represent every email message as a "vector" having the set (w1,w2,w3,...wN) of distinct words, we can calculate the probablity of spam with:
```
P(S|w1 ∩ w2 ∩... ∩ wN) = P(w1 ∩ w2 ∩...∩ wN|S)*P(S) / P(w1 ∩ w2 ∩...∩ wN) 
```
where *P(w1 ∩ w2 ∩ w3...∩ wN)* indicates the probability of **intersection** of all words "w".

If we (naively) assume that the probability of each word is independent of the probability of all other words (which is a rasonable assumption if we already factor in whether the email is spam or not) then we can simplify this to an approximate solution, taking the product of each word-probability expression:
```
P(S|w1 ∩ w2 ∩ ...∩ wN) = P(w1)*P(S) * P(w2)*P(S)*...*P(wN|S)*P(S) / P(w1)*P(w2)*...*P(wN)
```

If we encounter a word in the testing dataset which is not part of training dataset, P(w) will be 0 which will make the P(S|w) undefined (since we would have to divide by P(w). To address this issue we introduce **additive smoothing**. In additive smoothing we add a number N (typcially 1) to the numerator and add N times number of classes (words) to the denominator.

## Feature Engineering for Spam Detection:
Two common methods of feature-engineering for Text-Classification are:
  * **BOW**: Bag-of-Words
  * **TF-IDF**: Text-Frequency Inverse-Document-Frequency

## Naive-Bayes-Classification (NBC) with BOW: 
Text Classification using **BOW** for Naive-Bayes modeling uses simple word-counts (with additive-smoothing) to construct the known probabilities for the Naive-Bayes models.   

### Summarizing the NBC-BOW Text-Classification process:
**TRAINING THE ALGORITHM, CREATING THE MODEL**   
Iterate over the labelled spam/ham messages in the training set. For each word **w** in the set:  
    1) compute P(w|S): (N-spam-messages-containing-w)+1 / (N-spam-messages)+2
    2) compute P(w|S): (N-ham-messages-containing-w)+1 / (N-ham-messages)+2 
    3) Compute P(S) = (N-spam-messsages)/(total-messages)
    4) Compute P(H) = (N-ham-messages)/(total-messages)

**TESTING THE MODEL**   
Iterate over a set of (unlabelled) test messages. For each message: 
    1) Create a set {w1,...,wN} of the distinct words in the message.
    2) Compute P(S|w1 ∩ w2 ∩ ...∩ wN) =  P(w1)*P(S) * P(w2)*P(S)*...*P(wN|S)*P(S) / P(w1)*P(w2)*...*P(wN)
    3) Compute P(H|w1 ∩ w2 ∩...∩ wN) = P(w1)*P(H) * P(w2)*P(H)*...*P(wN|H)*P(H) / P(w1)*P(w2)*...*P(wN)
    4) Classify as SPAM|HAM by comparing 2 and 3.  
       Since we are comparing, we can get rid of the denominator and just see which is numerator is bigger. 
        * SPAM if (P(w1)*P(S))*(P(w2)*P(S))*...*P(wN|S)*P(S) > (P(w1)*P(H))*(P(w2)*P(H))*...*P(wN|H)*P(H)
        * HAM  if (P(w1)*P(S))*(P(w2)*P(S))*...*P(wN|S)*P(S) < (P(w1)*P(H))*(P(w2)*P(H))*...*P(wN|H)*P(H)

If one of the words in the testing set was never seen in the training set, this will set the entire probablity to zero. Even if this never occurs, these calculations involve multiplying a lot of small numbers together which can lead to an "underflow" of numerical precision. To address these issues it is common practice to use a *log-transform* of the probabilities, so that our classification becomes:
```
  * SPAM if log( (P(w1)*P(S))*(P(w2)*P(S))*...*P(wN|S)*P(S) ) > log( (P(w1)*P(H))*(P(w2)*P(H))*...*P(wN|H)*P(H) )
  * HAM  if log( (P(w1)*P(S))*(P(w2)*P(S))*...*P(wN|S)*P(S) ) < log( (P(w1)*P(H))*(P(w2)*P(H))*...*P(wN|H)*P(H) )
```

Since log(ab) = log(a) + log(b), our classfication determination becomes:
```
  * SPAM if log(P(w1)*P(S)) + log((P(w2)*P(S)) + ... log(P(wN|S)*P(S)) > log(P(w1)*P(H)) + log(P(w2)*P(H))+...+log(P(wN|H)*P(H))
  * HAM  if log(P(w1)*P(S)) + log((P(w2)*P(S)) + ... log(P(wN|S)*P(S)) < log(P(w1)*P(H)) + log(P(w2)*P(H))+...+log(P(wN|H)*P(H))
```

With **TF-IDF**, we are not directly transforming the probabilities of each word. Instead, we can think about it as transforming the documents. With BOW each word in each document counted as 1, whereas with TF-IDF the words in the documents are counted as their TF-IDF weight. We get the known probabilities for Naive Bayes by adding up the TF-IDF weights instead of simply counting the number of words.

## Poisoning
A downside of this type of Bayesian filtering in cases of more targeted spam is that spammers will start using words (or whole pieces of text) that will deliberately lower the probablity score. After prolonged training, these words might get associated with spam, which is called *poisoning*. 


---
## Model-2: Support Vector Machine
 An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on the side of the gap on which they fall.

We can learn what this distribution is from messages that were identified as spam and messages that were identified as not being spam (sometimes called ham). 

The objective of the learning ability is to reduce the number of false positives. As annoying as it is to receive a spam message, it is worse to not receive an important message because a word was used that triggered the filter. 
