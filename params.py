import xgboost as xgb
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
params = {'eta':[0.01,0.2],'min_child_weight':[1],'max_depth':[i for i in xrange(3,11)],'gamma':[0],'max_delta_step':[0],'subsample':[0.5,1],
		'colsample_bytree':[0.8,0.5,1],'colsample_bylevel':[1],'lambda':[1],'min_child_weight':[1]}


def best_params(train,target,params,model,objective = 'binary:logistic',n_estimators=100):
	gsearch1 = GridSearchCV(estimator = model( learning_rate =0.1, n_estimators=n_estimators, subsample=0.8, nthread=4, scale_pos_weight=1, seed=27), 
			 param_grid = params, scoring='roc_auc',n_jobs=1,iid=False, cv=5)
	gsearch1.fit(train,target)
	return gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
	
