from sklearn.svm import SVC
from plot import plot_learning_curve
import sys
from sklearn.metrics import classification_report
from sklearn.model_selection import ShuffleSplit


def svclassifier(xtrain, ytrain, xtest, ytest):

    print("Running SVM")
    svm = SVC(C=2.0, kernel='rbf', gamma=0.001)
    sys.stdout.flush()

    print("Plotting Learnign curve")
    sys.stdout.flush()
    cv = ShuffleSplit(n_splits=15, test_size=0.2, random_state=0)
    plot_learning_curve(svm, "SVM Training", xtrain, ytrain,
                        ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    svm.fit(xtrain, ytrain)

    svmpred = svm.predict(xtrain)
    print(classification_report(ytrain, svmpred))

    svmpred = svm.predict(xtest)
    print(classification_report(ytest, svmpred))

    return svmpred
