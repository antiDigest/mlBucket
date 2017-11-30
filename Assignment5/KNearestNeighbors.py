import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from CrossValidation import crossValidation


def knn(X, Y, X_test, Y_test, cv=10):
    clf = KNeighborsClassifier(n_neighbors=5, leaf_size=30, p=2)
    return crossValidation(clf, X, Y)


def knnclf():
    return KNeighborsClassifier(n_neighbors=5, leaf_size=30, p=2)
