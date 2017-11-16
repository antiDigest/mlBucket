import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def adaboost(X, Y, X_test, Y_test):

    clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.1)

    # clf.fit(X, Y)
    scores = cross_val_score(clf, X, Y, cv=8, scoring='accuracy')

    return scores.mean()
