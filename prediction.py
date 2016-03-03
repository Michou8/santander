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
for t in ctrain:
	tmp = {}
	target.append(float(t['TARGET']))
	for key in t:
		if key not in remove:
			tmp[key] = float(t[key])
	#print tmp
	train.append(tmp)
	print 'Size:\t',len(train)
	if len(train) == 5000 :
		"""print 'Done'
		dv = DictVectorizer()
		X = dv.fit_transform(train)
		y = target
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
		print('EST)')
		rf = RandomForestRegressor(n_estimators = 500,n_jobs=1)
		rf = rf.fit(X_train,y_train)
		#pred = rf.predict(X_test)"""
		break
		#raw_input()
dv = DictVectorizer()
X = dv.fit_transform(train)
del train
y = target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('EST)')
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
        #train.append(tmp)
        #if len(train) == 1000 :
        #dv = DictVectorizer()
        X = dv.transform([tmp])
                #y = target
                #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                #rf = RandomForestRegressor(n_estimators = 500)
                #rf = rf.fit(X_train,y_train)
        pred = rf.predict(X)[0]
	prediction.append(pred)	
	print (len(prediction)*100)/75819.0
	print len(prediction)
import pandas as pd
pd.DataFrame({'ID':id_test,'TARGET':prediction}).to_csv('sub-03-03-2016.csv',index=False)
