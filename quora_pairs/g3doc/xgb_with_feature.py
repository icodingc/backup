# coding: utf-8
import numpy as np
import pandas as pd
import datetime
import operator
from sklearn.cross_validation import train_test_split

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import sys
sys.path.append('/home/zhangxs/soft/xgboost/python-package/')
import xgboost as xgb

df_train = pd.read_csv("./../quora_pairs/data/new_feature/train_feature.csv")
x_trains = df_train.values
x_train = x_trains[:,2:]
print "train x",x_train.shape
df_train_src = pd.read_csv("./../quora_pairs/data/train.csv")
y_train = df_train_src["is_duplicate"].values
print "train_y",y_train.shape

RS = 1
ROUNDS = 500

print("Started")
np.random.seed(RS)

def train_xgb(X, y, params):
    print("Will train XGB for {} rounds, RandomSeed: {}".format(ROUNDS, RS))
    x, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=RS)

    xg_train = xgb.DMatrix(x, label=y_train)
    xg_val = xgb.DMatrix(X_val, label=y_val)

    watchlist  = [(xg_train,'train'), (xg_val,'eval')]
    return xgb.train(params, xg_train, ROUNDS, watchlist,early_stopping_rounds=50)

def predict_xgb(clr, X_test):
    return clr.predict(xgb.DMatrix(X_test))


# In[12]:

params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = ['logloss','error']
params['eta'] = 0.2
params['gpu_id'] = 0
params['updater'] = 'grow_gpu'
params['nthread']=8
params['min_child_weight']=3
params['max_depth'] = 4
params['subsample']=0.8
params['colsample_bytree']=0.8
params['silent'] = 1
params['seed'] = RS
print("Training")
bst = train_xgb(x_train, y_train, params)


# In[28]:

#get_ipython().magic(u'matplotlib inline')
#xgb.plot_importance(clr)


# ## For TEST

# In[29]:

df_test = pd.read_csv("./../quora_pairs/data/new_feature/test_feature.csv")
x_test = df_test.values
x_test = x_test[:,2:]
df_test_src = pd.read_csv("./../quora_pairs/data/test.csv")


preds = predict_xgb(clr, x_test)
print preds.shape
preds[:10]


print("Writing output...")
sub = pd.DataFrame()
sub['test_id'] = df_test_src['test_id']
sub['is_duplicate'] = preds *.75
sub.to_csv("xgb_with_feature_2000round_rs42.csv", index=False)

