import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from CrossValidation import crossValidation

def adaboost(X, Y, X_test, Y_test, cv=10):

    clf = AdaBoostClassifier(n_estimators=100, learning_rate=0.2)
    return crossValidation(clf, X, Y)
