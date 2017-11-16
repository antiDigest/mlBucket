import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold


def crossValidation(clf, X, Y, cv=10):

    skf = StratifiedKFold(n_splits=cv)

    skf.get_n_splits(X, Y)

    accuracy = 0
    recall = 0
    precision = 0

    for train_index, test_index in skf.split(X, Y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, xtest = X[train_index], X[test_index]
        y_train, ytest = Y[train_index], Y[test_index]

        clf.fit(X_train, y_train)

        Y_pred = clf.predict(xtest)

        accuracy += accuracy_score(ytest, Y_pred)
        recall += recall_score(ytest, Y_pred, average='weighted')
        precision += precision_score(ytest, Y_pred, average='weighted')

    return {"accuracy": accuracy / cv,
            "recall": recall / cv,
            'precision': precision / cv}
