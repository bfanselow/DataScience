# Data Science

### Collecting and "cleaning" gigantic quantities of (structured or unstructured) data to be processed by mathematical/statistical-analysis and machine-learning algorithms with the goal of extracting meaningful knowledge and insights (patterns, trends, predictions, etc.).

![DataScienceVenn](https://raw.githubusercontent.com/bfanselow/DataScience/master/data_science_venn.jpg)

Venn Diagram Source: [Drew Conway](http://drewconway.com/zia/2013/3/26/the-data-science-venn-diagram)

---

## Machine Learning Projects
 * **SHAM (Spam/Ham Analysis Engine)**: Simple Text-Classification using Naive-Bayes and SVM models to detect Spam.
 * **FakeNewsDetection**: Simplistic fake-news detector using PassiveAggressiveClassifier linear-regression model. 

Below is a dicussion of the basic Machine-Learning concpets, ML-Mathematics and the (Python) tools common to all of those projects. 
Each Machine-Learning project in the **ML_PROJECTS** folder has its own *README* file with a more in-depth discussion on the ML topics relevent to that project (background, algorithms, etc.). 

---
## Basic Data-Science Steps 
###  Not exactly this linear in practice - typcially more cycle/iterative
 1) Frame the problem: understaning business need. 
 2) Import Raw Data needed
 3) Format conversion/normalization: get all data into standard format for a single process to handle all remaining steps
 4) Rough Data Cleaning 
     * Handling Missing values
     * Handling Corrupted values
 5) Understand/explore the data
     * Pattern Detection
     * Distribution analysis 
     * Outlier Detection
 6) Fine Cleaning and Feature Engineering (perparing data for Machine-Learning algorithms)
      * Scaling and normalization: adjusting the range and center of data to ease learning and improve the interpretation of our results. 
      * Filling missing values: Real-world datasets often have missing values.
      * Feature selection: removing features because they are unimportant, redundant, or outright counterproductive to learning.
      * Feature coding: choosing a set of symbolic values to represent different categories.
      * Feature extraction: moving from low-level features that are unsuitable for learning (we get poor testing results) to higher-level features which are useful for learning.
 7) Predictive (ML) modelling: train and validate models.
 8) Interpreting results: visualiizing outcomes

---
## Machine Learning Mathematics 
A **"vector"** is a list of one or more ordinary numbers. It can be thought of as a coordinate in an N-dimesional space. Mathematically this (Euclidean vector) represents line from the "origin" of that space to that coordinate where the origin is the zero-point of all dimensions.

Machine learning uses **"feature vectors"** which can be thought of as a specific location (i.e. coordinate) in a N-dimensional "feature" space, where N is the number of unique attributes/features.  A feature-vector then is simply an array of N unique features.

Typcially, one of the very first steps in making a machine learning model is "vectorizing" the data. Under the hood, machine-le
arning algorithms perform mathematical computations on vectors and collections-of-vectors (a.k.a. **matrix**).

**Machine Leanring computational terms:**
 * **Scalar**: A single number.
 * **Vector** : A list of values.(tensor of rank 1)
 * **Matrix**: A two dimensional list of values.(tensor or rank 2)
 * **Tensor**: A multi dimensional matrix with rank n.

---
## (Python) Machine Learning Tools 

The following Python modules are used by the above ML projects:
 ## Modules from SciPy family
 * NumPy 
 * Pandas   
 * Matplotlib
 ## Built on top of SciPy
   * Sklearn 

**NumPy** and **Pandas** both provide fast and intuitive mathematical computation on arrays/vectors and matrices. While they seem somewhat similar, each module has unique functionalities and strengths. 

The **NumPy** module is mainly used for working with numerical data using (Numpy) *Array* objects. With Arrays, one can perform mathematical operations on multiple values in the Arrays at the same time, and also perform operations between different Arrays, similar to matrix operations.

The **Pandas** module is mainly used for working with tabular data. It allows us to work with data in table form, such as in CSV or SQL database formats. We can also create tables of our own, and edit or add columns or rows to tables. Pandas provides **DataFrames** and **Series** objects which are very useful for working with and analyzing tablular data.

The **Matplotlib** module is used for data visualization (drawing charts and graphs) which is extremely useful for better understanding of the data.

**Sklearn** is a Machine Learning library providing various *classification*, *regression* and *clustering* algorithms including Naive-Bayes, Support-Vector-Machines, Random-Forests, Gradient-Boosting, k-means and DBSCAN and is desinged to interoperate with the above numerical libs.
