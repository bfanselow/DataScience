# Data Science

### Collecting and "cleaning" gigantic quantities of (structured or unstructured) data to be processed by mathematical/statistical-analysis and machine-learning algorithms with the goal of extracting meaningful knowledge and insights (patterns, trends, predictions, etc.).

<img src="https://github.com/bfanselow/DataScience/blob/master/img/data_science_venn.jpg" width="424" height="238">

Venn Diagram Source: [Drew Conway](http://drewconway.com/zia/2013/3/26/the-data-science-venn-diagram)
---

### Data science is a process that uses names and numbers to answer such questions, typically one of five types:

* How much or how many? (regression)
* Which category? (classification)
* Which group? (clustering)
* Is this weird? (anomaly detection)
* Which option should be taken? (recommendation)

### Machine Learning problems can be categorized into three major types:

#### Supervised
Supervised machine learning is the process of building a model that is capable of making predictions after being trained by repetition. Learning requires large amounts of tagged/labeled input data known as "training data".  The model learns from the labeled input.  Once trained, the model can make either (*classification* or *regression*) predictions on a new data set. With *classification* the model predicts discrete responses (a *label*), such as spam vs. ham email. With *regression* the machine predicts continuous responses (a *value*).  The more input data the model can learn from (i.e. more training!), the better it will be able to predict correct responses.

Supervised learning typically involves simpler algorithms and can provide very definitive results. However, it has two major drawbacks:
  1) Lots of labeled data is required for a well trained model - what if you don't have labeled data? What if the data is associated with an entirely novel situation?
  2) Even if you do have a large set of labelled data, the quality of the model's output is only as good as the quality of the input data - how do you know that the input data is accurately labeled?  
```
  Garbage-in --> Model --> Garbage-out
```

### Unsupervised
Unsupervised learning, by contrast, does not have *labelled* input data.  Without prior knowledge of data patterns, the model **discovers** patterns or density distributions in un-labeled input data - typically for the purpose of identifying clusters and associations in the data. As with supervised-learning, the more input data the machine can process, the more accurate and useful the results will be.

### Reinforcement Learning
Reinforcement learning is used to train the model to make a sequence of decisions, learning to achieve a goal in a complex, uncertain environment using a feedback mechanism based on previous choices.  The model gets rewarded or penalties for the actions it performs. Its goal is to maximize the total reward. Without any prior training, it’s up to the model to figure out how to maximize the reward, beginning with totally random trials and gradually making more and more sophisticated decisions.


**NOTE:**  
There are other problem types - hybrids of these, and variations (i.e. inverse-reinforcement), but these are the big three. 

---

## Machine Learning Projects
 * **SHAM (Spam/Ham Analysis Engine)**: Simple Text-Classification using **Naive-Bayes** and **SVM** models to detect Spam.
 * **FakeNewsDetection**: Simplistic fake-news detector using a PassiveAggressiveClassifier (linear-regression) model. 

Below is a dicussion of the basic Machine-Learning concpets, ML-Mathematics and the (Python) tools common to all of those projects. 
Each Machine-Learning project in the **ML_PROJECTS** folder has its own *README* file with a more in-depth discussion on the ML topics relevent to that project (background, algorithms, etc.). 

---
## Basic Data-Science Steps 
 1) Frame the problem: understanding business need. 
 2) Import Raw Data needed
 3) Exploring/understanding the data (often with visual charts/graphs)
     * Pattern Detection
     * Distribution analysis 
     * Outlier Detection
     * Identification of missing or corrupted values 
 4) Data pre-processing: get all data into standard format for a single process to handle all remaining steps.
       * Filling missing values (real-world datasets usually have missing values).
       * Handling corrupted values.
 5) Feature Engineering (preparing data for Machine-Learning algorithms)
      * Feature selection: removing features because they are unimportant, redundant, or outright counterproductive to learning.
      * Feature coding: choosing a set of symbolic values to represent different categories.
      * Feature extraction: moving from low-level features that are unsuitable for learning (we get poor testing results) to higher-level features which are useful for learning.
      * Scaling and normalization: adjusting the range and center of data to ease learning and improve the interpretation of our results. 
 6) Machine-Learning - predictive modeling (cyclic iterations over 3 steps).
      * Train the algorithm to create a ML model
      * Test the model's performace 
      * Improve model performance with algorithm tuning. 
 7) Interpreting results to extract meaning.

---
## Machine-Learning common mathematical terms
A **"vector"** is a list of one or more ordinary numbers. It can be thought of as a coordinate in an N-dimesional space. Mathematically this (Euclidean vector) represents line from the "origin" of that space to that coordinate where the origin is the zero-point of all dimensions.

Machine learning uses **"feature vectors"** which can be thought of as a specific location (i.e. coordinate) in a N-dimensional "feature" space, where N is the number of unique attributes/features.  A feature-vector then is simply an array of N unique features.

Typcially, one of the very first steps in making a machine learning model is "vectorizing" the data. Under the hood, machine-le
arning algorithms perform mathematical computations on vectors and collections-of-vectors (a.k.a. **matrix**).

**Machine Learning computational terms:**
 * **Scalar**: A single number.
 * **Vector** : A list of values (tensor of rank 1).
 * **Matrix**: A two dimensional list of values (tensor or rank 2).
 * **Tensor**: A multi dimensional matrix with rank **N**.

---
## (Python) Data-Science and Machine-Learning Tools 

The following Python modules are used by the above ML projects:
 * NumPy 
 * Pandas   
 * Matplotlib
 * Sklearn 
 * Nltk

**NumPy** and **Pandas** and **Matplotlib** are all part of the **SciPy** family of modules.

**NumPy** and **Pandas** both provide fast and intuitive mathematical computation on arrays/vectors and matrices. While they seem somewhat similar, each module has unique functionalities and strengths. 

The **NumPy** module is mainly used for working with numerical data using (Numpy) *Array* objects. With Arrays, one can perform mathematical operations on multiple values in the Arrays at the same time, and also perform operations between different Arrays, similar to matrix operations.

The **Pandas** module is mainly used for working with tabular data. It allows us to work with data in table form, such as in CSV or SQL database formats. We can also create tables of our own, and edit or add columns or rows to tables. Pandas provides **DataFrames** and **Series** objects which are very useful for working with and analyzing tablular data.

The **Matplotlib** module is used for data visualization (drawing charts and graphs) which is extremely useful for better understanding of the data.

**Sklearn** is a Machine Learning library built on top of SciPy, which provides various *classification*, *regression* and *clustering* algorithms including Naive-Bayes, Support-Vector-Machines, Random-Forests, Gradient-Boosting, k-means and DBSCAN and is desinged to interoperate with the above numerical libs.
 
**Nltk** (Natural Language Toolkit) is used specifically for data-science problems involving **text** data. It provides various text processing and cleaning tasks as tokenization, lemmatization, stemming, parsing, POS-tagging, etc.

