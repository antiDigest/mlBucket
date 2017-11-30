import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import scale, normalize, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

from DecisionTree import decisionTree, decisionTreeclf
from SVM import svc, svclf
from LogisticRegression import logisticRegression, lrclf
from Perceptron import perceptron, perceptronclf
from NeuralNetwork import neuralNet, nnclf
from Bagging import bagging, baggingclf
from GradientBoosting import gradientBoost, gradientBoostclf
from KNearestNeighbors import knn, knnclf
from RandomForest import randomForest, rfclf
from AdaBoost import adaboost, adaboostclf
from DeepLearning import deep, deepclf
from naivebayes import mnb, mnbclf


X = pd.read_csv("data/Crowdsourced Mapping/training.csv", header=0)
Y = X["class"]
X = X[list(X.columns)[1:]]
# print X.columns
X = SelectKBest(f_classif, k=20).fit_transform(X, Y)
# print X.columns
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
# X = normalize(X)

X_test = pd.read_csv("data/Crowdsourced Mapping/testing.csv", header=0)
Y_test = X_test["class"]
X_test = X_test[list(X_test.columns)[1:]]
X_test = scale(X_test)

# print "Decision Tree: ", decisionTree(X, Y, X_test, Y_test)
# print "Perceptron: ", perceptron(X, Y, X_test, Y_test)
# print "Neural Network: ", neuralNet(X, Y, X_test, Y_test)
# print "Deep Learning: ", deep(X, Y, X_test, Y_test)
# print "SVM: ", svc(X, Y, X_test, Y_test)
# print "Naive Bayes: ", mnb(X, Y, X_test, Y_test)
# print "Logistic Regression: ", logisticRegression(X, Y, X_test, Y_test)
# print "KNearestNeighbors: ", knn(X, Y, X_test, Y_test)
# print "Random Forest: ", randomForest(X, Y, X_test, Y_test)
# print "Bagging: ", bagging(X, Y, X_test, Y_test)
# print "AdaBoost: ", adaboost(X, Y, X_test, Y_test)
# print "Gradient Boosting: ", gradientBoost(X, Y, X_test, Y_test)


classifiers = [adaboostclf, baggingclf, decisionTreeclf, deepclf,
               gradientBoostclf, knnclf, lrclf, mnbclf, nnclf, perceptronclf,
               rfclf, svclf]


def runAll(X, Y, cv=10):

    skf = StratifiedKFold(n_splits=cv)

    skf.get_n_splits(X, Y)

    count = 0
    accuracy = [0.] * len(classifiers)
    f1 = [0.] * len(classifiers)
    precision = [0.] * len(classifiers)

    for train_index, test_index in skf.split(X, Y):
        count += 1
        print "FOLD: ", count
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, xtest = X[train_index], X[test_index]
        y_train, ytest = Y[train_index], Y[test_index]

        for i, clf in enumerate(classifiers):
            clf = clf()
            clf.fit(X_train, y_train)

            Y_pred = clf.predict(xtest)

            accuracy[i] += accuracy_score(ytest, Y_pred)
            f1[i] += f1_score(ytest, Y_pred, average='weighted')
            precision[i] += precision_score(ytest, Y_pred, average='weighted')

    for i, clf in enumerate(classifiers):
        print str(clf) + ":", {"accuracy": accuracy[i] / cv,
                               "f1_score": f1[i] / cv,
                               'precision': precision[i] / cv}

    return


runAll(X, Y)
