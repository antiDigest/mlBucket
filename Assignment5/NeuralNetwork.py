import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def neuralNet(X, Y, X_test, Y_test):

    clf = MLPClassifier(hidden_layer_sizes=(
        30, 15, 1), learning_rate="invscaling")

    # clf.fit(X, Y)
    scores = cross_val_score(clf, X, Y, cv=8, scoring='accuracy')

    return scores.mean()
