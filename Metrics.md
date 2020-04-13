##  Machine-Learning Metrics

###  CONFUSION-MATRIX:
``` 
                       PREDICTED 
           +---------------------------------+ 
           |          | Positive  | Negative | 
           +----------+-----------+----------|
    ACTUAL | Positive |    TP     |    FN    |
           +----------+-----------+----------|
           | Negative |    FP     |    TN    |
           +---------------------------------+
``` 

  * **True Positive (TP)** — an outcome where the model correctly predicts the positive class.
  * **True Negative (TN)** - an outcome where the model correctly predicts the negative class.
  * **False Positive (FP)** - an outcome where the model incorrectly predicts the positive class.
  * **False Negative (FN)** - an outcome where the model incorrectly predicts the negative class.

##  METRIC TERMS: 

### ACCURACY
Model accuracy (in classification models) can be defined as the ratio of correctly classified samples to the total number of total samples:
```
  ACCURACY = (TP+TN)/(TP+TN+FP+FN)
```
Accuracy alone doesn’t tell the full story when we’re working with a class-imbalanced dataset where there’s a significant disparity between the number of positive and negative labels. For example, suppose we are trying to build a model to predict whether the patient has cancer or not.  Let’s assume we have a training dataset with 100 cases, 10 labeled as "Cancer", 90 labeled as "Normal".  Suppose our model gives the  the following results:   
  - TP (Actual Cancer and predicted Cancer) = 1
  - TN (Actual Normal and predicted Normal) = 90
  - FN (Actual Cancer and predicted Normal) = 8
  - FP (Actual Normal and predicted Cancer) = 1   

The accuracy of this model calculates to 91%. But is this model really that useful, even being so accurate? This model isn’t able to predict the actual cancer patients, which can have worst of all consequences.

### PRECISION
In a classification task, the precision for a class is the number of true positives divided by the total number of elements labeled as belonging to the positive class.
```
  PRECISION = (TP)/(TP+FP)
```
High precision means that an algorithm returned substantially more relevant results than irrelevant ones.  Our cancer model has a precision value of 0.5 — in other words, when it predicts cancer, it’s correct 50% of the time.

### RECALL
In this context, recall is defined as the number of true positives divided by the total number of elements that actually belong to the positive class (i.e. the sum of true positives and false negatives, which are items which were not labeled as belonging to the positive class but should have been).
```
  RECALL = (TP)/(TP+FN)
```
High recall means that an algorithm returned most of the relevant results. Our cancer model has a recall value of 0.11 — in other words, it correctly identifies only 11% of all cancer patients.

To fully evaluate the effectiveness of a model, it’s necessary to examine both precision and recall. Unfortunately, precision and recall are often in conflict. That is, improving precision typically reduces recall and vice versa. This is where F1-Score come in.

### F1-SCORE
The F1 score is the harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.  Since the harmonic mean of a list of numbers skews strongly toward the least elements of the list, it tends (compared to the arithmetic mean) to mitigate the impact of large outliers and aggravate the impact of small ones.
```
  F1-SCORE = 2*(precision)*(recall) / (precision + recall)
```
An F1 score punishes extreme values more. Ideally, an F1 Score could be an effective evaluation metric in the following classification scenarios:
   * When FP and FN are equally costly—meaning they miss on true positives or find false positives— both impact the model almost the same way, as in our cancer detection classification example
   * Adding more data doesn’t effectively change the outcome effectively
   * TN is high (like with cancel predictions, flood predictions, etc.)

