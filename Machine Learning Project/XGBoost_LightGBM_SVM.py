
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as gbm
from sklearn.svm import SVC


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Getting Data
train = pd.read_csv("train_set.csv")
train_labels = pd.read_csv("train_set_labels.csv")

train = pd.merge(train, train_labels, on='id')


# In[3]:


train.head()


# In[4]:


columns = list(set(list(train.columns)[1:-1]) - set(['num_private', 'basin',
                                                     'lga', 'region', 'funder',
                                                     'installer', 'scheme_name',
                                                     'scheme_management']))
X = train[columns]
Y = train[list(train.columns)[-1]]

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction import DictVectorizer

from sklearn.preprocessing import LabelEncoder

X_p = X
lb_make = LabelEncoder()
for column in columns:
    if X[column].dtype not in ['int64', 'float64']:
        X_p[column] = X[column].astype('category')
        X_p[column] = lb_make.fit_transform(X[column].fillna(method='ffill'))

Y_p = lb_make.fit_transform(Y)
# ohe = OneHotEncoder()
# ohe.fit(X['funder'])
# ohe.transform(X['funder']).head()
# X.head()
# cols = list(set(columns) - set(['amount_tsh', 'gps_height', 'longitude',
# 'latitude', 'region_code', 'district_code', 'population']))
# X_p = pd.get_dummies(X)


# In[5]:


# from sklearn.feature_selection import SelectKBest
# skb = SelectKBest(k = 10000)
# skb.fit(X_p, Y)
# X_p = skb.transform(X_p, Y)
X_p.shape
X_p.head()


# In[10]:


# TODO
pca = PCA(n_components=18)
X_p = pca.fit_transform(X_p, Y)
# y = pca.fit_transform(Y_p)
# y.shape
X_p.shape


# In[11]:


from sklearn.metrics import classification_report
from xgboost import XGBClassifier

xtrain, xtest, ytrain, ytest = train_test_split(X_p, Y_p, test_size=0.2)


# In[21]:


x = XGBClassifier(max_depth=18, learning_rate=0.03,
                  n_estimators=400, gamma=3, nthread=6)
x.fit(xtrain, ytrain)


# In[22]:


pred = x.predict(xtrain)

print(classification_report(ytrain, pred))

pred = x.predict(xtest)


print(classification_report(ytest, pred))


# In[23]:


from sklearn.metrics import accuracy_score, precision_score

print(accuracy_score(ytest, pred))
print(precision_score(ytest, pred, average='weighted'))


# In[43]:


from lightgbm import LGBMClassifier

lgbm = LGBMClassifier(boosting_type='dart', num_leaves=173, max_depth=14,
                      learning_rate=0.2, objective='multiclass', n_estimators=350, n_jobs=6)
lgbm.fit(xtrain, ytrain)


# In[44]:


lgbmpred = lgbm.predict(xtrain)

print(classification_report(ytrain, lgbmpred))

lgbmpred = lgbm.predict(xtest)

print(classification_report(ytest, lgbmpred))


# In[45]:


print(accuracy_score(ytest, lgbmpred))
print(precision_score(ytest, lgbmpred, average='weighted'))


# In[ ]:


svm = SVC(kernel='poly', degree=2, verbose=True)

svm.fit(xtrain, ytrain)


# In[23]:


svmpred = svm.predict(xtrain)

print(classification_report(ytrain, svmpred))

svmpred = svm.predict(xtest)

print(classification_report(ytest, svmpred))


# In[ ]:


print(accuracy_score(ytest, svmpred))
print(precision_score(ytest, svmpred, average='weighted'))
