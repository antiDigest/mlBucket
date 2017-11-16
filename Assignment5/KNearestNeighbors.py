import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def knn(X, Y, X_test, Y_test):

    clf = KNeighborsClassifier(n_neighbors=3, leaf_size=10, p=2)
    # clf.fit(X, Y)
    scores = cross_val_score(clf, X, Y, cv=8, scoring='accuracy')

    return scores.mean()
