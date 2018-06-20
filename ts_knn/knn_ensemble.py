# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from copy import copy, deepcopy
from scipy.ndimage.interpolation import shift
import pandas as pd

#def return_x(series, i, x):
#    x_columns = [series.name +'_t_'+ '{0:0>2}'.format(str(i))]
#    x_data = [list(series.shift((i)).values)]
#    x_df = pd.DataFrame(index = series.index, data = {k:v for k,v in zip(x_columns, x_data)}).dropna()
#    x = pd.concat([x,x_df], axis = 1).dropna()
#    return x


def get_data(series, steps, forward = False):
    if forward:
        fb = -1
    else:
        fb = 1
    columns = [series.name + '_t_'+ '{0:0>2}'.format(str(i)) for i in range(1,steps+1)]
    data = [list(series.shift(i*fb).values) for i in range(1,steps+1)]
    df = pd.DataFrame(index = series.index, data = {k:v for k,v in zip(columns, data)}).dropna()
    return df

def get_data_df(df, steps, forward = False):
    df_list = []
    for column in df.columns:
        d = get_data(df[column], steps, forward = forward)
        df_list.append(d)
    df = pd.concat(df_list, axis = 1)
    return df

#def get_y(endogenous, freqstr, limit, steps_ahead):
#        if isinstance(endogenous, pd.DataFrame):
#            if endogenous.shape[1]>1:
#                raise ValueError('Endogenous must be of shape (n,1)')
#            else:
#                endogenous=pd.Series(endogenous.iloc[:,0])
#        endog = endogenous.asfreq(freq = freqstr).interpolate(limit = limit).dropna()
#        y = get_data(endog, steps_ahead, forward = True)
#        y = y.dropna()
#        y_columns = [endog.name +'_t_'+ '{0:0>2}'.format(str(i)) for i in range(1,steps_ahead+1)]
#        return (y,y_columns,endog)


    
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
            
            
#    def fit(self, X, y, new_fit = True):
#        if new_fit:
#            self.X = X
#            self.y = y
#        shape = np.shape(X)
#        self.n = shape[0]
#        try:
#            self.params = shape[1]
#        except IndexError:
#            self.params = 1
#        for n in self.n_neighbors[0]:
#            self.model_dict[n]['model'] = self.model_dict[n]['model'].fit(X,y)
    
    def fit(self, x, y, freqstr, h, lags, limit = 5, new_fit = True):
        x = pd.DataFrame(x).asfreq(freq = freqstr).interpolate(limit = limit).dropna()
        if isinstance(y, pd.DataFrame):
            if y.shape[1]>1:
                raise ValueError('y must be of shape (n,1)')
        else:
            y = pd.DataFrame(y).asfreq(freq = freqstr).interpolate(limit = limit).dropna()
        X = get_data_df(x, lags, forward = False)
        Y = get_data_df(y, h, forward = True)
        idx = [x for x in X.index if x in Y.index]
        X = X.loc[idx]
        Y = Y.loc[idx]
        if new_fit:
            self.X = X
            self.y = Y
        for n in self.n_neighbors[0]:
            self.model_dict[n]['model'] = self.model_dict[n]['model'].fit(X,Y)
        
    
    
    
    
    
    
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
#    
#    def dynamic(self, X):
#        steps = len(list(self.y.columns))
#        Y_hat = []
#        self.fit(self.X,self.y.iloc[:,0:1], new_fit = False)
#        for x in X.values:
#            data = np.array(x).reshape(1,len(x))
#            y_hat = []
#            for step in range(steps):
#                y1 = self.static(data)[0][0]
#                data = shift(data,1, cval = y1)
#                y_hat.append(y1)
#            Y_hat.append(y_hat)
#        self.fit(self.X,self.y)
#        return(Y_hat)
    
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
        return rmse
          

    def forward_selection(self, endogenous, freqstr, h = 24, exogenous = None, lags = 5, interpolate = True, limit = 5, brk_at_min=False):
        """
        Enter endogenous and optional exogenous
        automatically format into the specified steps
        go through forward backward variable selection
        
        """

        
        y,y_columns,endog=get_y(endogenous, freqstr, limit, h)
        if isinstance(exogenous, pd.DataFrame) or isinstance(exogenous, pd.Series):
            exog = pd.DataFrame(exogenous.asfreq(freq = freqstr).interpolate(limit = limit).dropna())
        else:
            exog = None
        x = pd.DataFrame(index = endog.index)
        errors = {}
        min_error = float('inf')
        for i in range(1,lags+1):
            if isinstance(exog, pd.DataFrame):
                for column in exog.columns:
                    x = return_x(exog[column], i, x)
            else:
                x = return_x(endog, i, x)
            x = x.loc[y.index]
            grouped = x.groupby(pd.Grouper(freq = 'Y'))
            error_list = []
            for g,v in grouped:
                year = g.year
                x_test = v.dropna()
                y_test = y.loc[x_test.index]
                x_train = x[(x.index> str(year+1)) | (x.index<str(year))].dropna()
                y_train = y.loc[x_train.index]
                self.fit(x_train, y_train)
                err = self.error(x_test,y_test)
                error_list.append(err)
            s = pd.Series(index = y_columns, data = np.mean(error_list,axis = 0))
            errors.update({'step_'+str(i):s})
            error_df = pd.DataFrame(errors)
            error = error_df.mean().min()
            if brk_at_min:
                if min_error>error:
                    min_error = error
                else:
                    break
        return (y, error_df)
    
    def backward_selection(self, endogenous, freqstr, steps_ahead = 24, exogenous = None, steps = 5, interpolate = True, limit = 5, brk_at_min=False):

        y,y_columns,endog=get_y(endogenous, freqstr, limit, steps_ahead)
        if isinstance(exogenous, pd.DataFrame) or isinstance(exogenous, pd.Series):
            exog = pd.DataFrame(exogenous.asfreq(freq = freqstr).interpolate(limit = limit).dropna())
            x = get_data_df(exog, steps)
        else:
            exog = None
            x = get_data(endog, steps)
        min_error = float('inf')
        x = x.loc[y.index]
        errors = {}
        columns = x.columns
        max_step = max([int(x.split('_')[-1]) for x in columns])
        min_step = min([int(x.split('_')[-1]) for x in columns])
        for i in range(min_step, max_step+1):
            cols = [x for x in columns if int(x.split('_')[-1]) >= i]
            grouped = x.groupby(pd.Grouper(freq = 'Y'))
            error_list = []
            for g,v in grouped:
                year = g.year
                x_test = v.dropna()[cols]
                y_test = y.loc[x_test.index]
                x_train = x[(x.index> str(year+1)) | (x.index<str(year))].dropna()[cols]
                y_train = y.loc[x_train.index]
                self.fit(x_train, y_train)
                err = self.error(x_test,y_test)
                error_list.append(err)
            s = pd.Series(index = y_columns, data = np.mean(error_list,axis = 0))
            errors.update({'step_'+str(i):s})
            error_df = pd.DataFrame(errors)
            error = error_df.mean().min()
            if min_error>error:
                    min_error = error
                    c = cols
        return (error_df, c)
    

    def forward_backward_selection(self, endogenous, freqstr, steps_ahead = 24, 
                                   exogenous = None, max_steps = 5, 
                                   interpolate = True, limit = 5, 
                                   brk_at_min=False, intervals = True, p = .95):
        y,error_df = self.forward_selection(endogenous=endogenous, 
                                             freqstr=freqstr, 
                                             steps_ahead = steps_ahead, 
                                             exogenous = exogenous, 
                                             max_steps = max_steps, 
                                             interpolate = interpolate, 
                                             limit = limit, 
                                             brk_at_min=brk_at_min)
        
        steps = int(error_df.mean().idxmin().split('_')[-1])
        
        error_df, cols = self.backward_selection(endogenous=endogenous, 
                                                  freqstr=freqstr, 
                                                  steps_ahead = steps_ahead, 
                                                  exogenous = exogenous, 
                                                  steps = steps, 
                                                  interpolate = interpolate, 
                                                  limit = limit, 
                                                  brk_at_min=brk_at_min)
        
        y,_,endog=get_y(endogenous=endogenous, freqstr=freqstr, 
                        steps_ahead = steps_ahead, limit = limit)
                        
        if isinstance(exogenous, pd.DataFrame) or isinstance(exogenous, pd.Series):
            exog = pd.DataFrame(exogenous.asfreq(freq = freqstr).interpolate(limit = limit).dropna())
            x = get_data_df(exog, steps)
        else:
            exog = None
            x = get_data(endog, steps)
        x = x[cols].dropna()
        y=y.dropna()
        idx = [dx for dx in y.index if dx in x.index]
        y = y.loc[idx]
        x = x.loc[idx]
        
        if intervals:
            grouped = x.groupby(pd.Grouper(freq = 'Y'))
            error_list = []
            grps = len(grouped.groups)
            boots = int(1000/grps)
            for g,v in grouped:
                year = g.year
                x_test = v.dropna()
                y_test = y.loc[x_test.index]
                x_train = x[(x.index> str(year+1)) | (x.index<str(year))].dropna()
                y_train = y.loc[x_train.index]
                for i in range(boots):
                    x_sample_train = x_train.sample(frac = 1, replace = True)
                    y_sample_train = y_train.loc[x_sample_train.index]
                    model = KnnEnsemble()
                    model.fit(x_sample_train, y_sample_train)
                    preds = model.static(x_test)
                    e = y_test - preds
                    error_list.append(e)
            low = (1-p)/2
            high = 1-low        
            error = pd.concat(error_list, ignore_index = True)
            self.high = error.quantile(high).values
            self.low = error.quantile(low).values
        
        self.fit(x,y)
        
        return(error_df, x,y)
        
        
   