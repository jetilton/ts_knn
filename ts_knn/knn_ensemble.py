# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from copy import copy
from scipy.ndimage.interpolation import shift
import pandas as pd
class KnnEnsemble:
    
    def __init__(self, n_neighbors=[3,5,7], weights='uniform', algorithm='auto', 
                 leaf_size=30, p=2, metric='minkowski', metric_params=None, 
                 n_jobs=1, **kwargs):
        self.n_neighbors = n_neighbors,
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.model_dict = {}
        self.n = 0
        self.params = 0
        
        for n in self.n_neighbors[0]:
            model = KNeighborsRegressor(n_neighbors = n, weights=self.weights, algorithm=self.algorithm, 
                 leaf_size=self.leaf_size, p=self.p, metric=self.metric, metric_params=self.metric_params, 
                 n_jobs=self.n_jobs, **self.kwargs)
            
            self.model_dict.update({n:{'model':model}})
            
            
    def fit(self, X, y):
        shape = np.shape(X)
        self.n = shape[0]
        self.params = shape[1]
        for n in self.n_neighbors[0]:
            self.model_dict[n]['model'] = self.model_dict[n]['model'].fit(X,y)
            
    def predict(self, X):
        pred_list = []
        for n in self.n_neighbors[0]:
            try:
                self.model_dict[n]['predict'] = self.model_dict[n]['model'].predict(X)
            except KeyError:
                self.model_dict[n].update({'predict':self.model_dict[n]['model'].predict(X)})
            pred_list.append(self.model_dict[n]['predict'])
        return np.mean(pred_list, axis = 0)
    
    
    def aic(self, X, y, dynamic = False, **kwargs):
        if dynamic:
            y_hat = self.dynamic(X, kwargs['steps'])
        else:
            y_hat = self.predict(X)
        
        mse = np.mean(np.subtract(y,y_hat)**2)
        aic = self.n*np.log(mse) + 2 * (self.params +1)
        return {'aic':aic[0],'mse':mse[0]}
    
    
    def forward_selection(self, X_train,X_test, y_train,y_test):
        columns = X_train.columns
        r = max([int(x.split('_')[-1]) for x in columns])
        train_list =[]
        test_list = []
        step = 1
        results = {'step' : [], 'aic' : [], 'mse' : []}
        for i in range(r):
            cols = [x for x in columns if str(i+1) == x.split('_')[-1]]
            train = X_train[cols]
            train_list.append(train)
            test = X_test[cols]
            test_list.append(test)
            x_train = pd.concat(train_list, axis=1)
            x_test = pd.concat(test_list, axis=1)
            self.fit(x_train, y_train)
            mse = self.aic(x_test,y_test)['mse']
            aic = self.aic(x_test,y_test)['aic']
            step = i+1
            results['step'].append(step)
            results['mse'].append(mse)
            results['aic'].append(aic)
        return pd.DataFrame(results)
            
            
    
    
    def dynamic(self, X, steps):
        if steps > self.params-1:
            raise ValueError
        y_hat = []
        for x in X.values:
            data = np.array(x).reshape(1,len(x))
            for step in range(steps):
                y = self.predict(x.reshape(1,len(x)))[0]
                y_hat.append(y)
                data = shift(data,-1, cval = y)
        return np.array(y_hat).reshape(1,len(y_hat))
        
    
    
    
        



    
    