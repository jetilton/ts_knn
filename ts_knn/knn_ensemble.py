# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd


def return_alike_axis(X,Y):
    idx = [x for x in X.index if x in Y.index]
    X = X.loc[idx]
    Y = Y.loc[idx]
    return (X,Y)

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

def preprocess(data, y=False, freqstr='H', steps = 24, limit = 5, forward = False):
    if y:
        if isinstance(data, pd.DataFrame):
            if data.shape[1]>1:
                raise ValueError('y must be of shape (n,1)')
    else:
        data = pd.DataFrame(data).asfreq(freq = freqstr).interpolate(limit = limit).dropna()
    data = get_data_df(data, steps, forward = forward)
    
    return data

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
        self.lags = None
        self.x_shape = None
        self.freqstr = None
        self.limit = None
        self.x_test = None
        self.h = None
        
        
        for n in self.n_neighbors[0]:
            model = KNeighborsRegressor(n_neighbors = n, weights=self.weights, algorithm=self.algorithm, 
                 leaf_size=self.leaf_size, p=self.p, metric=self.metric, metric_params=self.metric_params, 
                 n_jobs=self.n_jobs, **self.kwargs)
            
            self.model_dict.update({n:{'model':model}})
            
    def fit(self, x, y, freqstr='H', h=24, lags=15, limit = 5, new_fit = True):
        if lags:
            self.freqstr = freqstr
            self.lags = lags
            self.limit = limit
            self.h = h
            x = pd.DataFrame(x).asfreq(freq = freqstr).interpolate(limit = limit).dropna()
            self.shape = x.shape
            if isinstance(y, pd.DataFrame):
                if y.shape[1]>1:
                    raise ValueError('y must be of shape (n,1)')
            else:
                y = pd.DataFrame(y).asfreq(freq = freqstr).interpolate(limit = limit).dropna()
            X = get_data_df(x, lags, forward = False)
            Y = get_data_df(y, h, forward = True)
            X,Y = return_alike_axis(X,Y)
        else:
            X=x
            Y=y
        if new_fit:
            self.X = X
            self.y = Y
        for n in self.n_neighbors[0]:
            self.model_dict[n]['model'] = self.model_dict[n]['model'].fit(X,Y)
        
    def static(self, X, test = False, reshape = True):
        if reshape:
            X = pd.DataFrame(X)
            X = X.asfreq(freq = self.freqstr).interpolate(limit = self.limit).dropna()
            X = get_data_df(X, self.lags, forward = False).dropna()
        if test:
            self.x_test = X
        pred_list = []
        for n in self.n_neighbors[0]:
            try:
                self.model_dict[n]['predict'] = self.model_dict[n]['model'].predict(X)
            except KeyError:
                self.model_dict[n].update({'predict':self.model_dict[n]['model'].predict(X)})
            pred_list.append(self.model_dict[n]['predict'])
        preds = np.mean(pred_list, axis = 0)
        return preds
    
    def predict(self, X, freqstr = 'H', h = 24):
        
        preds =  self.static(X, reshape = False)
        df_list = []
        idx = X.index
        date = ''
        for index,pred in enumerate(preds):
            if freqstr:
                ts = idx[index] 
                date= pd.date_range(ts, periods=len(pred), freq=freqstr) + 1
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
        y_test = pd.DataFrame(y_test).asfreq(freq = self.freqstr).interpolate(limit = self.limit).dropna()
        y_test = get_data_df(y_test, self.h, forward = True)
        X_test,y_test = return_alike_axis(X_test,y_test)
        if dynamic:
            y_hat = self.dynamic(X_test)
        else:
            y_hat = self.static(X_test)
        rmse = np.sqrt((np.subtract(y_test,y_hat)**2).mean())
        #aic = (np.log(rmse/self.n) + 2 * (self.params +1)).mean()
        return rmse
    
    def forward_selection(self, x_train, y_train, x_test, y_test, freqstr='H', h = 24, max_lags = 15, interpolate = True, limit = 5, brk_at_min=False):
        X_test = pd.DataFrame(x_test).asfreq(freq = freqstr).interpolate(limit = limit).dropna()
        Y_test = pd.DataFrame(y_test).asfreq(freq = freqstr).interpolate(limit = limit).dropna()
        Y_test = get_data_df(Y_test, h, forward = True).dropna()
        X_test,Y_test = return_alike_axis(X_test,Y_test)
        
        errors = {}
        min_rmse = float('inf')
        for lag in range(1,max_lags+1):
            self.fit(x_train, y_train, freqstr=freqstr, h=h, lags=lag, limit=limit, new_fit = True)
            x_test = get_data_df(X_test, lag, forward = False)
            x_test, y_test = return_alike_axis(x_test,Y_test)
            y_hat = self.static(x_test, test = False, reshape = False)
            rmse = np.sqrt((np.subtract(Y_test[lag:],y_hat)**2).mean())
            errors.update({'lag_'+str(lag):rmse})
            if brk_at_min:
                if rmse.mean()<min_rmse:
                    min_rmse = rmse.mean()
                else:
                    break
        return pd.DataFrame(data = errors)
    
    def backward_selection(self, x_train, y_train, x_test, y_test, freqstr='H', h = 24, lags = 15, interpolate = True, limit = 5, brk_at_min=False):
        """
        Given an x_train, y_train and x_test, y_test the backward selection removes the beginning lags and records the error
        returns the error dataframe
        """
        X_test = pd.DataFrame(x_test).asfreq(freq = freqstr).interpolate(limit = limit).dropna()
        Y_test = pd.DataFrame(y_test).asfreq(freq = freqstr).interpolate(limit = limit).dropna()
        x_test = get_data_df(X_test, lags, forward = False)
        y_test = get_data_df(Y_test, h, forward = True)
        x_test,y_test = return_alike_axis(x_test,y_test)
        self.fit(x_train, y_train, freqstr=freqstr, h=h, lags=lags, limit=limit, new_fit = True)
        columns = self.X.columns
        errors = {}
        
        for lag in range(1,lags+1):
            cols = [x for x in columns if int(x.split('_')[-1]) >= lag]
            self.fit(self.X[cols],self.y, lags = False)
            y_hat = self.static(x_test[cols], test = False, reshape = False)            
            rmse = np.sqrt((np.subtract(y_test,y_hat)**2).mean())
            errors.update({'lag_'+str(lag):rmse})
        errors_df = pd.DataFrame(errors)
        return errors_df
    
    
    def forward_backward_selection(self, x_train, y_train, x_test, y_test, freqstr='H', h = 24, max_lags = 15, interpolate = True, limit = 5, brk_at_min=True):
        
        forward_errors = self.forward_selection(x_train, y_train, x_test, 
                                                y_test, freqstr=freqstr, h = h, 
                                                max_lags = max_lags, 
                                                interpolate = interpolate, 
                                                limit = limit, 
                                                brk_at_min=brk_at_min)
        
        lags = int(forward_errors.mean().idxmin().split('_')[-1])
        
        backward_errors = self.backward_selection(x_train, y_train, 
                                                  x_test, y_test, 
                                                  freqstr=freqstr, h = h, 
                                                  lags = lags, 
                                                  interpolate = interpolate, 
                                                  limit = limit, 
                                                  brk_at_min=brk_at_min)
        min_lag = int(backward_errors.mean().idxmin().split('_')[-1])
        
        x_train = preprocess(x_train, steps=lags, limit = limit)
        y_train = preprocess(y_train, steps=h, limit = limit)
        x_train, y_train = return_alike_axis(x_train, y_train)
        x_train = x_train.iloc[:,min_lag-1:]
        y_train = y_train.iloc[:,min_lag-1:]
        
        self.fit(x_train,y_train, lags = False)
        
        
        return backward_errors
    def __rtrn_fwd_lags(self, endogenous, exogenous=None, offset='Y', freqstr='H', h = 24, max_lags = 15, interpolate = True, limit = 5, brk_at_min=False):
        if isinstance(exogenous, pd.DataFrame) or isinstance(exogenous, pd.Series):
            endogenous,exogenous =  return_alike_axis(endogenous,exogenous)
            
        end_grpd = endogenous.groupby(pd.Grouper(freq = offset))
        error_dict = {}
        i=0
        #Determine error in each offset for variale selection (Forward selection process)
        for g,v in end_grpd:
            x_test = v.dropna()
            x_train = endogenous.copy().drop(index = x_test.index)
            y_test = v.dropna()
            y_train = x_train.copy()
            if isinstance(exogenous, pd.DataFrame) or isinstance(exogenous, pd.Series):
                x_test = pd.concat([x_test, exogenous.loc[x_test.index]], axis = 1).dropna()
                x_train = pd.concat([x_train, exogenous.loc[x_train.index]], axis = 1).dropna()
            x_train, y_train = return_alike_axis(x_train,y_train)
            x_test, y_test = return_alike_axis(x_test,y_test)
            errors = self.forward_selection(x_train, y_train, x_test, y_test, 
                                       freqstr=freqstr, h = h, 
                                       max_lags = max_lags, 
                                       interpolate = interpolate, limit = limit, 
                                       brk_at_min=False)
            
            error_dict.update({'offset_'+str(i):errors.mean()})
            i+=1
        df = pd.DataFrame(error_dict)
        # do not use last offset to determine lags
        lags = int(df.iloc[:,:-1].mean(axis = 1).idxmin().split('_')[-1])
        return lags
    
    def __rtrn_bck_lag(self, endogenous, fwd_lags, exogenous=None, offset='Y', freqstr='H', h = 24, interpolate = True, limit = 5, brk_at_min=False):
        #backward selection process
        end_grpd = endogenous.groupby(pd.Grouper(freq = offset))
        error_dict = {}
        i=0
        for g,v in end_grpd:
            x_test = v.dropna()
            x_train = endogenous.copy().drop(index = x_test.index)
            y_test = v.dropna()
            y_train = x_train.copy()
            if isinstance(exogenous, pd.DataFrame) or isinstance(exogenous, pd.Series):
                x_test = pd.concat([x_test, exogenous.loc[x_test.index]], axis = 1).dropna()
                x_train = pd.concat([x_train, exogenous.loc[x_train.index]], axis = 1).dropna()
            x_train, y_train = return_alike_axis(x_train,y_train)
            x_test, y_test = return_alike_axis(x_test,y_test)
            errors = self.backward_selection(x_train, y_train, x_test, y_test, 
                                        freqstr=freqstr, h = h, lags = fwd_lags, 
                                        interpolate = interpolate, 
                                        limit = limit, brk_at_min=False)
            error_dict.update({'offset_'+str(i):errors.mean()})
            i+=1
        df = pd.DataFrame(error_dict)
        #do not use last offset to determine lag
        lag = int(df.iloc[:,:-1].mean(axis = 1).idxmin().split('_')[-1])
        return lag
    
    
    def automatic(self, endogenous, exogenous=None, offset='Y', freqstr='H', h = 24, max_lags = 15, start_time = None, interpolate = True, limit = 5, brk_at_min=False):
        """
        gets the data and breaks it into the designated sasonality
        goes through forward selection averages values
        Get lowest value for forward selection go through backward selection
        get prediction interval
        fit model without last season
        return prediction with intervals (this will be a new class eventually)
        
        """

        fwd_lags = self.__rtrn_fwd_lags(endogenous=endogenous, exogenous=exogenous, offset=offset, freqstr=freqstr, h = h, max_lags = max_lags, interpolate = interpolate, limit = limit, brk_at_min=brk_at_min)
        back_lag = self.__rtrn_bck_lag(endogenous=endogenous, fwd_lags=fwd_lags, exogenous=exogenous, offset=offset, freqstr=freqstr, h = h,  interpolate = interpolate, limit = limit, brk_at_min=brk_at_min)
        

        end_grpd = endogenous.groupby(pd.Grouper(freq = offset))
        last_offset = [v for g,v in end_grpd][-1].dropna()
        x_train = endogenous.drop(index = last_offset.index)
        y_train =  get_data_df(x_train.copy(), h, forward = True)
        if isinstance(exogenous, pd.DataFrame) or isinstance(exogenous, pd.Series):
            x_train = pd.concat([x_train, exogenous.loc[x_train.index]], axis = 1).dropna()
        x_train = get_data_df(x_train, fwd_lags, forward = False).iloc[:,back_lag-1:]
        x_train, y_train = return_alike_axis(x_train,y_train)
        self.fit(x_train, y_train, lags = False)
        
        x_test = get_data_df(last_offset, fwd_lags, forward = False).iloc[:,back_lag-1:]
        final_x = x_test.iloc[-1,:].values.reshape(1,len(x_test.columns))
        x_predict = pd.DataFrame(data=final_x, columns = x_test.columns, index = [x_test.iloc[-1,:].name])
        y_test = get_data_df(last_offset.copy(), h, forward = True)
        x_test, y_test = return_alike_axis(x_test,y_test)
        y_hat = self.static(x_test, test = False, reshape = False)            
        rmse = np.sqrt((np.subtract(y_test,y_hat)**2).mean())
        y_hat_final = self.static(x_predict, test = False, reshape = False)   
        return y_hat_final,rmse
      

   