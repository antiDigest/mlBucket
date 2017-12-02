# from lightgbm import LGBMClassifier
import lightgbm as lgb
from plot import plot_learning_curve
import sys
from sklearn.metrics import classification_report
from sklearn.model_selection import ShuffleSplit


def light(xtrain, ytrain, xtest, ytest):
    print("Running LightGBM")
    lgbm = lgb.LGBMClassifier(boosting_type='dart', num_leaves=173, max_depth=14,
                              learning_rate=0.2, objective='multiclass', n_estimators=350, n_jobs=6)
    sys.stdout.flush()
    print("Plotting Learnign curve")
    sys.stdout.flush()
    cv = ShuffleSplit(n_splits=15, test_size=0.2, random_state=0)
    plot_learning_curve(lgbm, "DART Classifier Training", xtrain, ytrain,
                        ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    lgbm.fit(xtrain, ytrain)

    lgbmpred = lgbm.predict(xtrain)
    print(classification_report(ytrain, lgbmpred))

    lgbmpred = lgbm.predict(xtest)
    print(classification_report(ytest, lgbmpred))

    return lgbmpred
