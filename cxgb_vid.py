import pandas as pd
import numpy as np
import xgboost as xgb
import json
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
with open('stats.json') as f:
        remove_ = json.load(f)
remove = remove_['removes']
train = pd.read_csv('train.csv')[:5000]
train.drop(remove,axis=1)
print train.info()
target = 'TARGET'
IDCol = 'ID'
test = pd.read_csv('test.csv')
predictors = [x for x in train.columns if x not in ['TARGET', 'ID']]
"""xgb1 = XGBClassifier(
 learning_rate =0.1,
 n_estimators=1000,
 max_depth=5,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)
modelfit(xgb1, train, predictors)"""
param_test1 = {
 'max_depth':range(3,10,2),
 'min_child_weight':range(1,6,2)
}
import params
model = xgb.XGBRegressor
a,b,c = params.best_params(train[predictors],train[target],param_test1,model,objective = 'binary:logistic',n_estimators=100)
print a
print b
print c
