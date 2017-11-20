
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as gbm
import sklearn.svm as SVC


# In[2]:

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Getting Data
train = pd.read_csv("train_set.csv")
train_labels = pd.read_csv("train_set_labels.csv")

train = pd.merge(train, train_labels, on='id')
train, test = train_test_split(train, test_size=0.2)


# In[4]:

train.head()


# In[5]:

test.head()


# In[91]:

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
# cols = list(set(columns) - set(['amount_tsh', 'gps_height', 'longitude', 'latitude', 'region_code', 'district_code', 'population']))
# X_p = pd.get_dummies(X)


# In[ ]:




# In[51]:

# from sklearn.feature_selection import SelectKBest
# skb = SelectKBest(k = 10000)
# skb.fit(X_p, Y)
# X_p = skb.transform(X_p, Y)
X_p.shape
X_p.head()


# In[93]:

# TODO
pca = PCA(n_components=18)
X_p = pca.fit_transform(X_p, Y)
# y = pca.fit_transform(Y_p)
# y.shape
X_p.shape


# In[57]:

xgb_x = xgb.DMatrix(X_p)


# In[94]:

dtrain = xgb.DMatrix(X_p, label=Y_p)


# In[76]:

print(dtrain)


# In[95]:

param = {'max_depth': 2, 'eta': 1, 'silent': 1}
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['eval_metric'] = ['auc', 'ams@0']
bst = xgb.train(params=param, dtrain=dtrain)


# In[83]:

bst.predict(X_p)

