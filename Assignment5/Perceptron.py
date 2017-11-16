import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from CrossValidation import crossValidation


def perceptron(X, Y, X_test, Y_test, cv=10):

    clf = Perceptron(penalty='elasticnet', alpha=0.0001)
    return crossValidation(clf, X, Y)
