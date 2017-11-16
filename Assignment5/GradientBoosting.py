import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from CrossValidation import crossValidation


def gradientBoost(X, Y, X_test, Y_test, cv=10):

    clf = GradientBoostingClassifier(learning_rate=1, n_estimators=50, max_depth=4)
    return crossValidation(clf, X, Y)

