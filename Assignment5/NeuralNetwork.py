import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold
from CrossValidation import crossValidation


def neuralNet(X, Y, X_test, Y_test, cv=10):

    clf = MLPClassifier(hidden_layer_sizes=(
        20, 10, 5, 2, 1), learning_rate="adaptive")

    return crossValidation(clf, X, Y)
