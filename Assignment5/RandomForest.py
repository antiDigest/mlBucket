import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from CrossValidation import crossValidation


def randomForest(X, Y, X_test, Y_test, cv=10):

    clf = RandomForestClassifier(
        n_estimators=150, max_depth=None, max_leaf_nodes=None)
    return crossValidation(clf, X, Y)


def rfclf():

    return RandomForestClassifier(
        n_estimators=150, max_depth=None, max_leaf_nodes=None)
