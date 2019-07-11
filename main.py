
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

import numpy as np


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')
# the training data folder must be passed as first argument
dataset = load_files("/Users/tanay/Desktop/news_file/", description= None, categories= None, load_content = True, encoding='latin1', decode_error='strict', shuffle= False, random_state=42)
print("n_samples: %d" % len(dataset.data))

# split the dataset in training and test set:
docs_train, docs_test, y_train, y_test = train_test_split(
    dataset.data, dataset.target, test_size=0.25, random_state=None)

# TASK: Build a vectorizer / classifier pipeline that filters out tokens
# that are too rare or too frequent
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(dataset.data)
X_train_counts.shape

text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
text_clf.fit(dataset.data, dataset.target)

# TASK: Build a grid search to find out whether unigrams or bigrams are
# more useful.
parameters = {
'vect__ngram_range': [(1, 1), (1, 2)],
'tfidf__use_idf': (True, False),
'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, iid=False, n_jobs=-1)
# Fit the pipeline on the training set using grid search for the parameters
gs_clf = gs_clf.fit(dataset.data[:400], dataset.target[:400])

# TASK: print the cross-validated scores for the each parameters set
# explored by the grid search
gs_clf.best_score_

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
# TASK: Predict the outcome on the testing set and store it in a variable
# named y_predicted
y_test = dataset.target
y_predicted = text_clf.predict(dataset.data)

# Print the classification report
print(metrics.classification_report(y_test, y_predicted, target_names=dataset.target_names))
f = open("/Users/tanay/Desktop/important\ files/sklearn_tut_workspace/skeletons/test.txt","r")
testdata = """
 'Strategic options' for free-to-air TV under review 
 Updated

The Federal Government has confirmed it is reviewing whether new technology can allow television broadcast to remain free to viewers.

Almost all Australians who receive their free-to-air television via cable, satellite or fibre-optic lines receive a signal at a cost of around $75 per month.

The Convergence Review has recommended that pay TV operators, cable companies and Internet service providers receive a fee to license their signals for free.

This could offer the $75 subscription fee to be applied to someone who gets the service via the internet.

"One of the recommendations made is that there should be a single compulsory right that anyone can access over the internet," Minister for Communications Stephen Conroy said.

"There are many ways that one could engage in one's own local radio service or local television service or access a video-on-demand platform.

"But this would be the opportunity for somebody to pay for their free-to-air viewing on a subscription model or free-to-air viewing on a licence fee model.

"They would get the benefit of a more effective competitor who would provide the same service to their customers."

Mr Conroy says other stakeholders, including unions, are also raising concerns about whether infrastructure upgrades would be required to support the changes.

"The Government is going to ensure that people who currently receive their free-to-air television in that way will continue to get their free-to-air television free of charge," he said.

"But also it's important that no infrastructure is lost."

The review by the Convergence Review group, led by former Telstra chief Sol Trujillo, also recommended a cap on advertising on all local broadcasters' digital channels to keep costs down.

Saying that time had come for change, the report said "the commercial channels' business models are broken".

"As the report discusses today, the market which provides free-to-air digital television - from both competing providers, such as major subscription digital TV, broadcast and associated online businesses as well as direct competitors - is insufficiently competitive to maintain competition or to produce the benefits of strong market demand, sustainable investment, efficiency and innovation in free-to-air," it said.

But the report found that even the large number of viewers in major markets such as Sydney and Melbourne have more than 30 channels in which to choose from.

From next year, the three networks that currently broadcast to most viewers through free-to-air channels - Nine, Ten and Seven - will switch to digital channels.

ABC/AAP

Topics: television-broadcasting, business-economics-and-finance, industry, broadcasting, information-and-communication, internet-culture, government-and-politics, federal-government, australia

First posted
"""
for text_clf in text_clfs:
    try:
        text_clf.predict(testdata)
    except NotFittedError as e:
        print(repr(e))
# Print and plot the confusion matrix
cm = metrics.confusion_matrix(y_test, y_predicted)
print(cm)

plt.matshow(cm)
plt.show()
