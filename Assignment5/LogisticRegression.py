import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from CrossValidation import crossValidation


def logisticRegression(X, Y, X_test, Y_test, cv=10):

    clf = LogisticRegression(penalty="l1", C=2.0)
    return crossValidation(clf, X, Y)

