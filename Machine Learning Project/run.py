# Necessary
import pandas as pd
import numpy as np
import argparse
import sys

# Model Selection
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score

# Encoders
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

# Classifiers
from svm import svclassifier
from dart import light
from xgbalgo import xgb

# Getting Data
print("Getting Data")
train = pd.read_csv("train_set.csv")
train_labels = pd.read_csv("train_set_labels.csv")

train = pd.merge(train, train_labels, on='id')
sys.stdout.flush()

columns = list(set(list(train.columns)[1:-1]) - set(['num_private', 'basin',
                                                     'lga', 'region', 'funder',
                                                     'installer', 'scheme_name',
                                                     'scheme_management']))

X = train[columns]
Y = train[list(train.columns)[-1]]


X_p = X
lb_make = LabelEncoder()
for column in columns:
    if X[column].dtype not in ['int64', 'float64']:
        X_p[column] = X[column].astype('category')
        X_p[column] = lb_make.fit_transform(X[column].fillna(method='ffill'))

Y_p = lb_make.fit_transform(Y)

print("Principal Component Analysis")
pca = PCA(n_components=20)
X_p = pca.fit_transform(X_p, Y)
sys.stdout.flush()

print("Splitting Data")
xtrain, xtest, ytrain, ytest = train_test_split(X_p, Y_p, test_size=0.2)
sys.stdout.flush()

# Argument to switch between predictions
parser = argparse.ArgumentParser(description='What descriptor to run.')
parser.add_argument('value', metavar='N', type=int,
                    help='(0: xgboost, 1: lightgbm, 2: svm)')

args = parser.parse_args()

print("Now to algorithms")
sys.stdout.flush()
if(args.value == 0):
    pred = xgb(xtrain, ytrain, xtest, ytest)
elif args.value == 1:
    pred = light(xtrain, ytrain, xtest, ytest)
else:
    pred = svclassifier(xtrain, ytrain, xtest, ytest)

print("Accuracy and Precision")
print(accuracy_score(ytest, pred))
print(precision_score(ytest, pred, average='weighted'))
sys.stdout.flush()
