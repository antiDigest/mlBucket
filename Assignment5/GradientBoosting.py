import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def gradientBoost(X, Y, X_test, Y_test):

    clf = GradientBoostingClassifier(learning_rate=1, n_estimators=50, max_depth=4)

    # clf.fit(X, Y)
    scores = cross_val_score(clf, X, Y, cv=8, scoring='accuracy')

    return scores.mean()
