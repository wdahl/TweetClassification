from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pylab as pl
import numpy as np
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
import json
import pickle

# Load from file
pkl_filename = "pickle_model.pkl"  
with open(pkl_filename, 'rb') as file:  
    clf = pickle.load(file)

training_model = json.loads(open("vocab.txt", "r").read())
vocab = training_model['vocab']

# predict the class labels of new tweets
tweets = []
for line in open('test_terms.txt').readlines():
    tweets.append(line)

# Generate X for testing tweets
X = []
for text in tweets:
    x = [0] * len(vocab)
    terms = [term for term in text.split() if len(term) > 2]
    for term in terms:
        if vocab.has_key(term):
            x[vocab[term]] += 1
    X.append(x)
y = clf.predict(X)

f = open("predictions.txt", "w")
length = len(open("test_tweets.txt").readlines())
with open("test_tweets.txt", "r") as test:
	f.write("{")
	i = 0
	for line in test.readlines():
		tweet = json.loads(line)
		f.write("\"")
		f.write(tweet['embersId'])
		f.write("\": ")
		if y[i] == 0:
			f.write("true")

		else:
			f.write("false")

		if i < length-1:
			f.write(", ")

		i += 1

	f.write("}")