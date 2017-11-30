import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from CrossValidation import crossValidation


def svc(X, Y, X_test, Y_test, cv=10):

    clf = SVC(kernel='linear', C=0.4, degree=2)
    return crossValidation(clf, X, Y)


def svclf():

    return SVC(kernel='linear', C=0.4, degree=2)
