import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import scale, normalize

from DecisionTree import decisionTree
from SVM import svc
from LogisticRegression import logisticRegression
from Perceptron import perceptron
from NeuralNetwork import neuralNet
from Bagging import bagging
from GradientBoosting import gradientBoost
from KNearestNeighbors import knn
from RandomForest import randomForest
from AdaBoost import adaboost


X = pd.read_csv("data/Crowdsourced Mapping/training.csv", header=0)
Y = X["class"]
X = X[list(X.columns)[1:]]
# print X.columns
X = SelectKBest(f_classif, k=20).fit_transform(X, Y)
# print X.columns
X = scale(X)
X = normalize(X)

X_test = pd.read_csv("data/Crowdsourced Mapping/testing.csv", header=0)
Y_test = X_test["class"]
X_test = X_test[list(X_test.columns)[1:]]
X_test = scale(X_test)

print "Decision Tree: ", decisionTree(X, Y, X_test, Y_test)
print "Perceptron: ", perceptron(X, Y, X_test, Y_test)
print "Neural Network: ", neuralNet(X, Y, X_test, Y_test)
print "SVM: ", svc(X, Y, X_test, Y_test)
print "Logistic Regression: ", logisticRegression(X, Y, X_test, Y_test)
print "KNearestNeighbors: ", knn(X, Y, X_test, Y_test)
print "Random Forest: ", randomForest(X, Y, X_test, Y_test)
print "Bagging: ", bagging(X, Y, X_test, Y_test)
print "AdaBoost: ", adaboost(X, Y, X_test, Y_test)
print "Gradient Boosting: ", gradientBoost(X, Y, X_test, Y_test)
