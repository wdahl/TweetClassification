Modules Installed:
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import pylab as pl
import numpy as np
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
import json
import pickle

How to Run files:
You Should be able to run the files by either pressing the run button or running form the terminal using python ./<filename.py>

You also need to have the file containing the postive tweets, negative tweets, and the test tweets in the same directory.

File Discrptions:
train.py trains the svm model. It takes a couple minutes but trust me it works I promise.
train.py will generate the file train_terms.txt, test_terms.txt, pickle_model.pkl, and vocab.txt. You dont need to do anythin with these files just make sure they are there when you run test.py

test.py tests the model using the test data and writes the results to the file called predictions.txt

train_terms.txt holds the terms extracted from the tweets in the triaining data.

test_terms.txt holds the testing terms extracted from the tweets in the testing data.

pickle_model.pkl holds the trained svm model so that you guys dont have to run the train.py code again whichi like a said takes A WHILE.

vocab.txt holds the words in vocab from the extracted tweets.

predicted.txt holds the embersId and the predicted truth value for the tweets in the test data.

