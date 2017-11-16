import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def perceptron(X, Y, X_test, Y_test):

    clf = Perceptron(penalty='elasticnet', alpha=0.0001)
    # clf.fit(X, Y)
    scores = cross_val_score(clf, X, Y, cv=8, scoring='accuracy')

    return scores.mean()
