"""Build a sentiment analysis / polarity model

Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess wether the opinion of the author is
positive or negative.

In this examples we will use a movie review dataset.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import logging
from optparse import OptionParser
import sys
from time import time
import foreshadow as fs
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError

import numpy as np


# NOTE: we put the following in a 'if __name__ == "__main__"' protected
# block to be able to use a multi-core grid search that also works under
# Windows, see: http://docs.python.org/library/multiprocessing.html#windows
# The multiprocessing module is used as the backend of joblib.Parallel
# that is used when n_jobs != 1 in GridSearchCV
# Display progress logs on stdout
def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')
# the training data folder must be passed as first argument
dataset = load_files("/Users/tanay/Desktop/news_file/", description= None, categories= None, load_content = True, encoding='latin1', decode_error='strict', shuffle= False, random_state=42)
print("n_samples: %d" % len(dataset.data))


# split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.25, random_state=None)
shadow = fs.Foreshadow()
shadow.fit(dataset.data, dataset.target)
shadow.score(X_test, y_test)
