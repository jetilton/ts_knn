# -*- coding: utf-8 -*-

import pandas as pd
from knn_ensemble import KnnEnsemble, get_data, get_data_df
import unittest
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from cwms_read.cwms_read import get_cwms
from datetime import timedelta

class TestMethods(unittest.TestCase):

    endogenous = pd.read_csv('endogenous.csv')
    endogenous['date'] = pd.to_datetime(endogenous['date'])
    endogenous.set_index('date', drop = True, inplace = True)
    exogenous = pd.read_csv('exogenous.csv')
    exogenous['date'] = pd.to_datetime(exogenous['date'])
    exogenous.set_index('date', drop = True, inplace = True)

#    def test_get_data_x(self):
#        x = self.endogenous
#        x = x.iloc[:,0] 
#        x.name = 'lag'
#        x = x.interpolate().dropna()
#        lag = np.random.randint(1,10,1)[0]
#        test1 = get_data(x, lag, forward = False)
#        test_index = np.random.randint(0,len(test1)-1,1)[0]
#        time_stamp = test1.index[test_index] 
#        vals = [x for x in reversed(test1.loc[time_stamp].values)]
#        index = pd.date_range(time_stamp- timedelta(hours = len(vals)), periods=len(vals), freq='H')
#        self.assertEqual(vals, x.loc[index].values.tolist())
#        
#    
#    def test_get_data_y(self):
#        x = self.endogenous
#        x = x.iloc[:,0] 
#        x.name = 'lag'
#        x = x.interpolate().dropna()
#        lag = np.random.randint(1,10,1)[0]
#        test1 = get_data(x, lag, forward = True)
#        test_index = np.random.randint(0,len(test1)-1,1)[0]
#        time_stamp = test1.index[test_index] 
#        vals = [x for x in test1.loc[time_stamp].values]
#        index = pd.date_range(time_stamp+ timedelta(hours = 1), periods=len(vals), freq='H')
#        self.assertEqual(vals, x.loc[index].values.tolist())
#        
#    def test_fit(self):
#        x = self.endogenous
#        x = x.iloc[:,0] 
#        x.name = 'lag'
#        lags = np.random.randint(1,10,1)[0]
#        model = KnnEnsemble()
#        model.fit(x, x, 'H', 24, lags, limit = 5, new_fit = True)
#        
#        self.assertEqual(False in model.X.index == model.y.index, False)
#        
#    def test_static(self):
#        x = self.endogenous
#        x = x.iloc[:,0] 
#        index = x.index[int(len(x)*.85)]
#        x_train = x[:index]
#        x_test = x[index:]
#        lags = np.random.randint(1,10,1)[0]
#        model = KnnEnsemble()
#        model.fit(x_train,x_train, 'H', 24, lags, limit = 5, new_fit = True)
#        y_hat = model.static(x_test, test = True)
#        k3 = KNeighborsRegressor(n_neighbors = 3)
#        k5 = KNeighborsRegressor(n_neighbors = 5)
#        k7 = KNeighborsRegressor(n_neighbors = 7)
#        results = []
#        x_trn = model.X
#        y_trn = model.y
#        for m in [k3, k5, k7]:
#            m.fit(x_trn, y_trn)
#            results.append(m.predict(model.x_test))
#        results = np.mean(results, axis =0)
#        res = results == y_hat
#        self.assertEqual(False in res, False)
#        
#    def test_forward_selection(self):
#        x = self.endogenous
#        x = x.iloc[:,0] 
#        index = x.index[int(len(x)*.85)]
#        x_train = x[:index]
#        y_train = x_train.copy()
#        x_test = x[index:]
#        y_test = x_test.copy()
#        lags = 15
#        model = KnnEnsemble()
#        errors = model.forward_selection(x_train,y_train, x_test,y_test,'H', 24, max_lags=lags, limit = 5, brk_at_min = True)
#        errors.mean()
#        min_test = errors.mean()[-1]>errors.mean()[-2] and errors.mean()[-3]>errors.mean()[-2]
#        self.assertEqual(min_test, True)
#    
#    def test_forward_selection_start_time(self):
#        x = self.endogenous
#        x = x.iloc[:,0] 
#        index = x.index[int(len(x)*.85)]
#        x_train = x[:index]
#        y_train = x_train.copy()
#        x_test = x[index:]
#        y_test = x_test.copy()
#        lags = 60
#        model = KnnEnsemble()
#        errors = model.forward_selection(x_train,y_train, x_test,y_test,'H', 24, start_time = '08:00', max_lags=lags, limit = 5, brk_at_min = False)
#        self.assertEqual(errors.mean().min(), 1.5084587278235737)
#
#    def test_backward_selection(self):
#        x = self.endogenous
#        x = x.iloc[:,0] 
#        index = x.index[int(len(x)*.85)]
#        x_train = x[:index]
#        y_train = x_train
#        x_test = x[index:]
#        y_test = x_test
#        lags = 15
#        model = KnnEnsemble()
#        errors=model.backward_selection(x_train,y_train, x_test,y_test,'H', 24, lags=lags, limit = 5, brk_at_min = True)
#        errors_forward = model.forward_selection(x_train,y_train, x_test,y_test,'H', 24, max_lags=lags, limit = 5, brk_at_min = False)
#
#        self.assertEqual(errors.mean()['lag_1'], errors_forward.mean()['lag_15'])
#    def test_backward_selection_start_time(self):
#        x = self.endogenous
#        x = x.iloc[:,0] 
#        index = x.index[int(len(x)*.85)]
#        x_train = x[:index]
#        y_train = x_train.copy()
#        x_test = x[index:]
#        y_test = x_test.copy()
#        lags = 60
#        model = KnnEnsemble()
#        errors=model.backward_selection(x_train,y_train, x_test,y_test,'H', 24, lags=lags, start_time = '08:00', limit = 5, brk_at_min = True)
#        errors_forward = model.forward_selection(x_train,y_train, x_test,y_test,'H', 24, start_time = '08:00', max_lags=lags, limit = 5, brk_at_min = False)
#
#        self.assertEqual(errors.mean()['lag_1'], errors_forward.mean()['lag_60'])
#        
#    def test_forward_backward_selection(self):
#        x = self.endogenous
#        x = x.iloc[:,0] 
#        index = x.index[int(len(x)*.85)]
#        x_train = x[:index]
#        y_train = x_train
#        x_test = x[index:]
#        y_test = x_test
#        max_lags = 15
#        model = KnnEnsemble()
#        forward_errors = model.forward_selection(x_train,y_train, x_test,y_test,'H', 24, max_lags=max_lags, limit = 5, brk_at_min = True)
#        lags = int(forward_errors.mean().idxmin().split('_')[-1]) 
#        backward_errors=model.forward_backward_selection(x_train,y_train, x_test,y_test,'H', 24, max_lags=max_lags, limit = 5, brk_at_min = True)
#        min_lag = int(backward_errors.mean().idxmin().split('_')[-1])
#        self.assertEqual(model.X.shape[1] == lags-min_lag+1, True)
#        
#    def test_forward_backward_selection_start_time(self):
#        x = self.endogenous
#        x = x.iloc[:,0] 
#        index = x.index[int(len(x)*.85)]
#        x_train = x[:index]
#        y_train = x_train
#        x_test = x[index:]
#        y_test = x_test
#        max_lags = 15
#        model = KnnEnsemble()
#        forward_errors = model.forward_selection(x_train,y_train, x_test,y_test,'H', 24, max_lags=max_lags,start_time='08:00', limit = 5, brk_at_min = True)
#        lags = int(forward_errors.mean().idxmin().split('_')[-1]) 
#        backward_errors=model.forward_backward_selection(x_train,y_train, x_test,y_test,'H', 24, max_lags=max_lags,start_time='08:00',limit = 5, brk_at_min = True)
#        min_lag = int(backward_errors.mean().idxmin().split('_')[-1])
#        self.assertEqual(model.X.shape[1] == lags-min_lag+1, True)
#    
#    def test_rtrn_fwd_lags_bck_lag(self):
#        model = KnnEnsemble()
#        fwd_lags = model._KnnEnsemble__rtrn_fwd_lags(self.endogenous, exogenous=None, offset='Y', freqstr='H', h = 24, max_lags = 15, interpolate = True, limit = 5, brk_at_min=False)
#        self.assertEqual(isinstance(fwd_lags, int), True)
#        lag = model._KnnEnsemble__rtrn_bck_lag(self.endogenous, fwd_lags=fwd_lags,exogenous=None, offset='Y', freqstr='H', h = 24, interpolate = True, limit = 5, brk_at_min=False)
#        self.assertEqual(isinstance(lag, int), True)
        
    def test_automatic_intervals(self):
        
        model = KnnEnsemble()
        #x_test = model.automatic(endogenous=self.endogenous,exogenous=self.exogenous,offset='Y',freqstr='H', h=24, max_lags=max_lags, limit = 5, brk_at_min = True)
        #self.assertEqual(x_test.shape[1], 2)
        f,b=model.automatic(self.endogenous, exogenous=None, offset='Y', freqstr='H', h = 24, max_lags = 15, start_time = None, interpolate = True, limit = 5, brk_at_min=False, p = .95)
        print(f,b)
if __name__ == '__main__':
    unittest.main()
    
    
