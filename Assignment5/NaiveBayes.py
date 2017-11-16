import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from CrossValidation import crossValidation


def mnb(X, Y, X_test, Y_test, cv=10):

    clf = MultinomialNB()

    return crossValidation(clf, X, Y)
