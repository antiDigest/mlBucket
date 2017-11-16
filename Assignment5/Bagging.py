import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from CrossValidation import crossValidation


def bagging(X, Y, X_test, Y_test, cv=10):
    clf = BaggingClassifier()
    return crossValidation(clf, X, Y)
