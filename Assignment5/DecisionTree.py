import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from CrossValidation import crossValidation


def decisionTree(X, Y, X_test, Y_test, cv=10):

    clf = tree.DecisionTreeClassifier(
        max_depth=20, min_samples_split=25, max_leaf_nodes=100)

    return crossValidation(clf, X, Y)


def decisionTreeclf():

    return tree.DecisionTreeClassifier(max_depth=20, min_samples_split=25, max_leaf_nodes=100)
