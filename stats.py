import xgboost as xgb
import pandas as pd
import numpy as np
import json
#from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('train.csv')
#test = pd.read_csv('test.csv')
print(train.info())

remove = []
stats = {}
columns = train.columns
for col in train.columns:
    if train[col].std() == 0:
        remove.append(col)
for i in xrange(len(columns)):
	c_i = columns[i]
	stats[c_i] = dict(train[c_i].describe())
	stats[c_i]['unique'] = len(train[c_i].unique())
	for j in xrange(i+1,len(columns)):
		c_j = columns[j]
		if c_i not in remove and c_j not in remove and train[c_i].equals(train[c_j]):
			print c_j
			remove.append(c_j)
	print len(remove)
stats['removes'] = remove
train.drop(remove,axis=1)
with open('stats.json','wb') as f:
	json.dump(stats,f,indent=4)

print(train.info())

