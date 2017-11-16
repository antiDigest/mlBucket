import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def svc(X, Y, X_test, Y_test):

    clf = SVC(kernel='poly', C=1, degree=2)
    # clf.fit(X, Y)
    scores = cross_val_score(clf, X, Y, cv=8, scoring='accuracy')

    return scores.mean()
