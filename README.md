# Data Science

### Collecting and "cleaning" gigantic quantities of (structured or unstructured) data to be processed by mathematical/statistical-analysis and machine-learning algorithms with the goal of extracting meaningful knowledge and insights (patterns, trends, predictions, etc.).

![DataScienceVenn](https://raw.githubusercontent.com/bfanselow/DataScience/master/data_science_venn.jpg)

Venn Diagram Source: [Drew Conway](http://drewconway.com/zia/2013/3/26/the-data-science-venn-diagram)

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
## Machine Learning Tools (Python)
### SciPy
 * NumPy 
 * Pandas   
 * Matplotlib

**NumPy** and **Pandas** both provide fast and intuitive mathematical computation on arrays/vectors and matrices. While they seem somewhat similar, each module has unique functionalities and strengths. 

The **NumPy** module is mainly used for working with numerical data using (Numpy) *Array* objects. With Arrays, one can perform mathematical operations on multiple values in the Arrays at the same time, and also perform operations between different Arrays, similar to matrix operations.

The **Pandas** module is mainly used for working with tabular data. It allows us to work with data in table form, such as in CSV or SQL database formats. We can also create tables of our own, and edit or add columns or rows to tables. Pandas provides **DataFrames** and **Series** objects which are very useful for working with and analyzing tablular data.

The **Matplotlib** module is used for data visualization (drawing charts and graphs) which is extremely useful for better understanding of the data.
 
