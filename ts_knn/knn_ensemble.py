# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from copy import copy, deepcopy
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
        self.high = None
        self.low = None
    
        
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
        try:
            self.params = shape[1]
        except IndexError:
            self.params = 1
        for n in self.n_neighbors[0]:
            self.model_dict[n]['model'] = self.model_dict[n]['model'].fit(X,y)
            
    def static(self, X):
        pred_list = []
        for n in self.n_neighbors[0]:
            try:
                self.model_dict[n]['predict'] = self.model_dict[n]['model'].predict(X)
            except KeyError:
                self.model_dict[n].update({'predict':self.model_dict[n]['model'].predict(X)})
            pred_list.append(self.model_dict[n]['predict'])
        
        preds = np.mean(pred_list, axis = 0)
        
        return preds
    
    def dynamic(self, X):
        steps = len(list(self.y.columns))
        Y_hat = []
        self.fit(self.X,self.y.iloc[:,0:1], new_fit = False)
        for x in X.values:
            data = np.array(x).reshape(1,len(x))
            y_hat = []
            for step in range(steps):
                y1 = self.static(data)[0][0]
                data = shift(data,1, cval = y1)
                y_hat.append(y1)
            Y_hat.append(y_hat)
        self.fit(self.X,self.y)
        return(Y_hat)
    
    def predict(self, X, dynamic = False, freqstr = 'H', step = 1):
        if dynamic:
            preds =  self.dynamic(X)
        else:
            preds =  self.static(X)
        df_list = []
        idx = X.index
        date = ''
        for index,pred in enumerate(preds):
            if freqstr:
                ts = idx[index] 
                date= pd.date_range(ts, periods=len(pred), freq=freqstr) + step
            df = pd.DataFrame(data = pred, columns = ['y_hat'])
            if isinstance(date, pd.core.indexes.datetimes.DatetimeIndex):
                df['date'] = date
                df = df.set_index('date', drop = True)
            try:
                df['high']  = df['y_hat'] + self.high
                df['low']  = df['y_hat'] + self.low
            except:
                pass
            df_list.append(df)
        return df_list
    
    
    def error(self, X_test, y_test, dynamic = False):
        if dynamic:
            y_hat = self.dynamic(X_test)
        else:
            y_hat = self.static(X_test)
        rmse = np.sqrt((np.subtract(y_test,y_hat)**2).mean())
        #aic = (np.log(rmse/self.n) + 2 * (self.params +1)).mean()
        return {'rmse':rmse}
    
    
    def forward_selection(self, X_train,X_test, y_train,y_test, dynamic = False):
        columns = X_train.columns
        max_step = max([int(x.split('_')[-1]) for x in columns])
        min_step = min([int(x.split('_')[-1]) for x in columns])
        results = {'step' : [], 'rmse' : []}
        min_rmse = float('inf')
        df = False
        for step in range(min_step, max_step+1):
            cols = [x for x in columns if step >= int(x.split('_')[-1])]
            x_train = X_train[cols].copy()
            x_test = X_test[cols].copy()
            self.fit(x_train, y_train)
            error = self.error(x_test, y_test, dynamic = dynamic)
            rmse = error['rmse'].mean()
            #aic = aic_mse['aic']
            results['step'].append(step)
            results['rmse'].append(rmse)
            #results['aic'].append(aic)
            if rmse < min_rmse:
                min_rmse = rmse
                df = x_train.copy()
#            else:
#                return (pd.DataFrame(results), df)
        return (pd.DataFrame(results),df)
    
    def backward_selection(self, X_train,X_test, y_train,y_test):
        self.fit(X_train, y_train)
        x_test = X_test[list(X_train.columns)]
        error = self.error(x_test,y_test, dynamic = False)
        #min_aic = aic_mse['aic']
        min_rmse = error['rmse'].mean()
        columns = list(X_train.columns)
        max_step = max([int(x.split('_')[-1]) for x in columns])
        min_step = min([int(x.split('_')[-1]) for x in columns])
        df = X_train.copy()
        results = {'step' : [max_step], 'rmse' : [deepcopy(min_rmse)]}
        for i in range(min_step, max_step):
            columns = [x for x in columns if int(x.split('_')[-1]) > i]
            x_train = X_train[columns]
            x_test = X_test[columns]
            self.fit(x_train, y_train)
            error = self.error(x_test,y_test, dynamic = False)
            rmse = error['rmse'].mean()
            #error = aic_mse['aic']
            step = max_step - i
            results['step'].append(step)
            results['rmse'].append(rmse)
            #results['aic'].append(aic)
            if rmse < min_rmse:
                min_rmse = rmse
                df = x_train.copy()
        results = pd.DataFrame(results)
        self.fit(df, y_train)
        error = self.pred_intervals(X_test[df.columns],y_test)
        return (results,error,df)
    
    def forward_backward_selection(self, X_train,X_test, y_train,y_test):
        results_f,X_trn = self.forward_selection(X_train,X_test, y_train,y_test, dynamic = False)
        self.fit(X_trn, y_train)
        x_test = X_test[list(X_trn.columns)]
        #error = self.error(x_test,y_test, dynamic = False)
        #min_aic = aic_mse['aic']
        min_rmse = results_f['rmse'].min()
        columns = list(X_trn.columns)
        max_step = max([int(x.split('_')[-1]) for x in columns])
        min_step = min([int(x.split('_')[-1]) for x in columns])
        df = X_trn.copy()
        results = {'step' : [max_step], 'rmse' : [deepcopy(min_rmse)]}
        for i in range(min_step, max_step):
            columns = [x for x in columns if int(x.split('_')[-1]) > i]
            x_train = X_trn[columns]
            x_test = X_test[columns]
            self.fit(x_train, y_train)
            error = self.error(x_test,y_test, dynamic = False)
            rmse = error['rmse'].mean()
            #error = aic_mse['aic']
            step = max_step - i
            results['step'].append(step)
            results['rmse'].append(rmse)
            #results['aic'].append(aic)
            if rmse < min_rmse:
                min_rmse = rmse
                df = x_train.copy()
        results = pd.DataFrame(results)
        self.fit(df, y_train)
        error = self.pred_intervals(X_test[df.columns],y_test)
        return (results_f,results,error,df)
    
    def pred_intervals(self, X_test, y_test, p = .95):
        low = (1-p)/2
        high = 1-low
        x_train = self.X
        y_train = self.y
        error_list = []
        for i in range(1000):
            x_sample_train = x_train.sample(frac = 1, replace = True)
            y_sample_train = y_train.loc[x_sample_train.index]
            model = KnnEnsemble()
            model.fit(x_sample_train, y_sample_train)
            preds = model. static(X_test)
            e = y_test - preds
            error_list.append(e)
        error = pd.concat(error_list, ignore_index = True)
        self.high = error.quantile(high).values
        self.low = error.quantile(low).values
        return error

    def automatic(endogenous, exogenous = None, steps = 5):
        """
        Enter endogenous and optional exogenous
        automatically format into the specified steps
        go through forward backward variable selection
        
        """
        pass
    
    
    