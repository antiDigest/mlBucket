import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def decisionTree(X, Y, X_test, Y_test):

    clf = tree.DecisionTreeClassifier(
        max_depth=8, min_samples_split=5, max_leaf_nodes=65)

    # clf.fit(X, Y)
    scores = cross_val_score(clf, X, Y, cv=8, scoring='accuracy')

    return scores.mean()

    # Y_pred = clf.predict(X_test)

    # print accuracy_score(Y_test, Y_pred)
