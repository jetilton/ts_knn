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
        self.X = None
        self.y = None
    
        
        for n in self.n_neighbors[0]:
            model = KNeighborsRegressor(n_neighbors = n, weights=self.weights, algorithm=self.algorithm, 
                 leaf_size=self.leaf_size, p=self.p, metric=self.metric, metric_params=self.metric_params, 
                 n_jobs=self.n_jobs, **self.kwargs)
            
            self.model_dict.update({n:{'model':model}})
            
            
    def fit(self, X, y, new_fit = True):
        if new_fit:
            self.X = X
            self.y = y
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
    
    
    def aic(self, X_test, y_test, dynamic = False, **kwargs):
        if dynamic:
            y_hat = self.dynamic(X_test)
        else:
            y_hat = self.predict(X_test)
        mse = np.mean(np.sqrt(np.subtract(y_test,y_hat)**2)).sum()/len(list(y_test.columns))
        aic = (self.n*np.log(mse) + 2 * (self.params +1))/len(list(y_test.columns))
        return {'aic':aic,'mse':mse}
    
    def forward_selection(self, X_train,X_test, y_train,y_test, dynamic = False, **kwargs):
        columns = X_train.columns
        max_step = max([int(x.split('_')[-1]) for x in columns])
        min_step = min([int(x.split('_')[-1]) for x in columns])
        results = {'step' : [], 'aic' : [], 'mse' : []}
        min_aic = float('inf')
        df = False
        for step in range(min_step, max_step+1):
            cols = [x for x in columns if step >= int(x.split('_')[-1])]
            x_train = X_train[cols].copy()
            x_test = X_test[cols].copy()
            self.fit(x_train, y_train)
            aic_mse = self.aic(x_test, y_test, dynamic = dynamic, **kwargs)
            mse = aic_mse['mse']
            aic = aic_mse['aic']
            results['step'].append(step)
            results['mse'].append(mse)
            results['aic'].append(aic)
            if aic < min_aic:
                min_aic = aic
                df = x_train.copy()
        results = pd.DataFrame(results)
        return (results,df)
    
    def forward_backward_selection(self, X_train,X_test, y_train,y_test):
        results,X_trn = self.forward_selection(X_train,X_test, y_train,y_test, dynamic = False)
        self.fit(X_trn, y_train)
        x_test = X_test[list(X_trn.columns)]
        aic_mse = self.aic(x_test,y_test, dynamic = False)
        min_aic = aic_mse['aic']
        min_mse = aic_mse['mse']
        columns = list(X_trn.columns)
        df = X_trn.copy()
        results = {'step' : [len(columns)], 'aic' : [copy.deepcopy(min_aic)], 'mse' : [copy.deepcopy(min_mse)]}
        for i in range(len(columns)-1):
            columns.pop(0) 
            x_train = X_trn[columns]
            x_test = X_test[columns]
            self.fit(x_train, y_train)
            aic_mse = self.aic(x_test,y_test, dynamic = False)
            mse = aic_mse['mse']
            aic = aic_mse['aic']
            step = len(columns)
            results['step'].append(step)
            results['mse'].append(mse)
            results['aic'].append(aic)
            if aic < min_aic:
                min_aic = aic
                df = x_train.copy()

        results = pd.DataFrame(results)
        return (results,df)
            

    def dynamic(self, X):
        steps = len(list(self.y.columns))
        Y_hat = []
        self.fit(self.X,self.y.iloc[:,0:1], new_fit = False)
        for x in X.values:
            data = np.array(x).reshape(1,len(x))
            y_hat = []
            for step in range(steps):
                y1 = self.predict(data)[0][0]
                data = shift(data,1, cval = y1)
                y_hat.append(y1)
            Y_hat.append(y_hat)
        
        self.fit(self.X,self.y)
        return(Y_hat)