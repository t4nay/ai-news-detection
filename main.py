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
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import SGDClassifier
import numpy as np
from pandas import DataFrame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.collections import QuadMesh
import seaborn as sn
from sklearn import svm
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_score


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')
# the training data folder must be passed as first argument
dataset = load_files("/Users/tanay/Desktop/news_file/", description= None, categories= None, load_content = True, encoding='latin1', decode_error='strict', shuffle= False, random_state=42)



# split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.2, random_state=None)
print("n_samples: %d" % len(dataset.data))

# TASK: Build a vectorizer / classifier pipeline that filters out tokens
# that are too rare or too frequent
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(docs_train)
X_train_counts.shape

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
text_clf.fit(docs_train, y_train)

# TASK: Build a grid search to find out whether unigrams or bigrams are
# more useful.
parameters = {
'vect__ngram_range': [(1, 1), (1, 2)],
'tfidf__use_idf': (True, False),
'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=6, iid=False, n_jobs=-1)
# Fit the pipeline on the training set using grid search for the parameters
gs_clf = gs_clf.fit(docs_train[:6000], y_train[:6000])

# TASK: print the cross-validated scores for the each parameters set
# explored by the grid search
print (gs_clf.best_score_)

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
# TASK: Predict the outcome on the testing set and store it in a variable
# named y_predicted


y_predicted = text_clf.predict(docs_test)

# Print the classification report
print(metrics.classification_report(y_test, y_predicted, target_names=dataset.target_names))
scores = cross_val_score(text_clf, y_test, docs_test, cv=6)
print(scores);
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Print and plot the confusion matrix
cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)

plt.matshow(cm)
plt.show()
