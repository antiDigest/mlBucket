import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

clf = BaggingClassifier()

X = pd.read_csv("data/Crowdsourced Mapping/training.csv", header=0)
Y = X["class"]
X = X[list(X.columns)[1:]]

X_test = pd.read_csv("data/Crowdsourced Mapping/testing.csv", header=0)
Y_test = X_test["class"]
X_test = X_test[list(X_test.columns)[1:]]

# clf.fit(X, Y)

scores = cross_val_score(clf, X, Y, cv=10, scoring='accuracy')

print(scores.mean())

# Y_pred = clf.predict(X_test)

# print accuracy_score(Y_test, Y_pred)
