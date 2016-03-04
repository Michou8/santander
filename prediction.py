import xgboost as xgb
import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_curve
from sklearn.cross_validation import train_test_split
import time
import csv
with open('stats.json') as f:
        remove_ = json.load(f)
remove = remove_['removes']
ctrain = csv.DictReader(open('train.csv'))
remove.append('ID')
remove.append('TARGET')
id_train = []
target = []
train = []
ti = time.time()
for t in ctrain:
	tmp = {}
	target.append(float(t['TARGET']))
	for key in t:
		if key not in remove:
			tmp[key] = float(t[key])
	#print tmp
	train.append(tmp)
	if len(train) > 5000:
		print 'Size:\t',len(train)
		print time.time()-ti
		break
dv = DictVectorizer()
X = dv.fit_transform(train)
del train
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators = 500,n_jobs=1)
rf = rf.fit(X_train,y_train)
ctrain = csv.DictReader(open('test.csv'))
id_test = []
target = []
train = []
prediction = []
ti = time.time()
for t in ctrain:
        tmp = {}
	id_test.append(t['ID'])
        for key in t:
                if key not in remove:
                        tmp[key] = float(t[key])
        #print tmp
        train.append(tmp)

        if len(train) > 1000:
        	#dv = DictVectorizer()
	        X = dv.transform(train)
	        pred = rf.predict(X)
		for p in pred:
			prediction.append(p)
		print time.time()-ti
		ti = time.time()
		train = []
	#prediction.append(pred)	
	#print (len(prediction)*100)/75819.0
	#print len(prediction)
X = dv.transform(train)
pred = rf.predict(X)
for p in pred:
	prediction.append(p)

import pandas as pd
pd.DataFrame({'ID':id_test,'TARGET':prediction}).to_csv('sub-'+str(time.time())+'.csv',index=False)
