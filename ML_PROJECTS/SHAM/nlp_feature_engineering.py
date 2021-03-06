"""

  File: nlp_feature_engineering.py

  Description:
    Methods for NLP feature-engineering.
    These methods provide several options for tuning our model (ngrams, stopwords, etc.).
    Try tweaking some of these parameters and re-testing model performance
  
    For example, it was found that removing stopwords BEFORE n-gram compilation resulted
    in much lower performace!

  Requires:
    * pip install nltk
    * from Python shell download "punkt" tokenizer and english stopwords: 
        >>> import nltk
            nltk.download(): select "D"
                              ==> select "punkt" 
                              ==> select "stopwords" 

"""

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def message_to_feature_words(message, lower_case=True, stem=True, stop_words=True, ngram=2):
    """
      Take an input text string and translate it to a list of words while
      performing multiple (optional) NLTK word operations:
        * lower-case 
        * n-gram 
        * remove stop-words 
        * word-stemming
      Required args:
        * message (str): raw message text
        * **kwargs: feature-engineering options
      Return (list): cleaned/processed list of words
    """
    if lower_case:
        message = message.lower()

    ## Tokenizers divide strings into lists of substrings
    words = word_tokenize(message)
    N = len(words)
   
    ## Keep only if > 2 chars
    words = [w for w in words if len(w) > 2]
   
    ## removing stopwords BEFORE n-gram compile results in much lower performace!

    ## if N-gram, compile n-grams and return list
    if ngram > 1:
        words_ngram = []
        for i in range(len(words) - ngram + 1):
            words_ngram += [' '.join(words[i:i + ngram])]
        return(words_ngram) 
    
    ## remove STOP words
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    
    ## word stemming
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]

    return( words )
