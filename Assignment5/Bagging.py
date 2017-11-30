import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from CrossValidation import crossValidation


def bagging(X, Y, X_test, Y_test):
    clf = BaggingClassifier(
        n_estimators=100, max_features=0.5, bootstrap=False)

    return crossValidation(clf, X, Y)


def baggingclf():
    return BaggingClassifier(n_estimators=100, max_features=0.5, bootstrap=False)
