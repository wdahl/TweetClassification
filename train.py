__author__ = 'fengchen'

from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pylab as pl
import numpy as np
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
import json
import pickle

f = open("train_terms.txt", "w")
for line in open("train_posi_tweets_2017.txt", "r").readlines():
    tweet = json.loads(line)
    f.write("1,")
    for term in tweet['text_items'][0]:
        f.write(term)
        f.write(" ")

    f.write("\n")

for line in open("train_nega_tweets_2017.txt", "r").readlines():
    tweet = json.loads(line)
    f.write("0,")
    for term in tweet['text_items'][0]:
        f.write(term)
        f.write(" ")

    f.write("\n")

f.close()

f = open("test_terms.txt", "w")
for line in open("test_tweets.txt", "r").readlines():
    tweet = json.loads(line)
    for term in tweet['text_items'][0]:
        f.write(term)
        f.write(" ")

    f.write("\n")

f.close()

tweets = []
for line in open('train_terms.txt').readlines():
    items = line.split(',')
    tweets.append([int(items[0]), items[1].lower().strip()])

vocab = dict()
for class_label, text in tweets:
    for term in text.split():
        term = term.lower()
        if len(term) > 2:
            if vocab.has_key(term):
                vocab[term] = vocab[term] + 1
            else:
                vocab[term] = 1

# Remove terms whose frequencies are less than a threshold (e.g., 15)
vocab = {term: freq for term, freq in vocab.items() if freq > 20}
# Generate an id (starting from 0) for each term in vocab
vocab = {term: idx for idx, (term, freq) in enumerate(vocab.items())}
print len(vocab)

# Generate X and y
X = []
y = []
for class_label, text in tweets:
    x = [0] * len(vocab)
    terms = [term for term in text.split() if len(term) > 2]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1

    y.append(class_label)
    X.append(x)

# 10 folder cross validation to estimate the best w and b
svc = svm.LinearSVC()
Cs = range(1, 20)
clf = GridSearchCV(estimator=svc, param_grid=dict(C=Cs), cv = 10)
clf.fit(X, y)

# Save to file in the current working directory
pkl_filename = "pickle_model.pkl"  
with open(pkl_filename, 'wb') as file:  
    pickle.dump(clf, file)

f = open("vocab.txt", "w")
svm_dict = {"vocab": vocab}
svm_string = json.dumps(svm_dict)
f.write(svm_string)