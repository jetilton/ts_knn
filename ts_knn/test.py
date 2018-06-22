# -*- coding: utf-8 -*-

import pandas as pd
from knn_ensemble import KnnEnsemble, get_data, get_data_df
import unittest
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from cwms_read.cwms_read import get_cwms
from datetime import timedelta

x_train = pd.read_csv('x_train.csv', index_col = 0)[:21]
y_train = pd.read_csv('y_train.csv', index_col = 0)[:21]
x_test = pd.read_csv('x_test.csv', index_col = 0)[:6]
y_test = pd.read_csv('y_test.csv', index_col = 0)[:6]

#x_train = pd.DataFrame(data = [x[0] for x in np.linspace(0,100,11).reshape(11,1)])
#y_train = x_train * 5
#x_test = pd.DataFrame(data = [x[0] for x in np.array([1,5]).reshape(2,1)])
#y_test = x_test*5
#




class TestMethods(unittest.TestCase):
    start_date = (2007, 3, 1)
    end_date =  (2010, 4,27)
    end_date_exog =  (2011, 4,27)
    endogenous = get_cwms('TDA.%-Saturation-TDG.Inst.1Hour.0.GOES-COMPUTED-REV', start_date = start_date, end_date = end_date, public = True, fill = True)
    exogenous =  get_cwms('JHAW.%-Saturation-TDG.Inst.1Hour.0.GOES-COMPUTED-REV', start_date = start_date, end_date = end_date_exog, public = True, fill = True)


    def test_get_data_x(self):
        x = self.endogenous
        x = x.iloc[:,0] 
        x.name = 'lag'
        x = x.interpolate().dropna()
        lag = np.random.randint(1,10,1)[0]
        test1 = get_data(x, lag, forward = False)
        test_index = np.random.randint(0,len(test1)-1,1)[0]
        time_stamp = test1.index[test_index] 
        vals = [x for x in reversed(test1.loc[time_stamp].values)]
        index = pd.date_range(time_stamp- timedelta(hours = len(vals)), periods=len(vals), freq='H')
        self.assertEqual(vals, x.loc[index].values.tolist())
        
    
    def test_get_data_y(self):
        x = self.endogenous
        x = x.iloc[:,0] 
        x.name = 'lag'
        x = x.interpolate().dropna()
        lag = np.random.randint(1,10,1)[0]
        test1 = get_data(x, lag, forward = True)
        test_index = np.random.randint(0,len(test1)-1,1)[0]
        time_stamp = test1.index[test_index] 
        vals = [x for x in test1.loc[time_stamp].values]
        index = pd.date_range(time_stamp+ timedelta(hours = 1), periods=len(vals), freq='H')
        self.assertEqual(vals, x.loc[index].values.tolist())
        
    def test_fit(self):
        x = self.endogenous
        x = x.iloc[:,0] 
        x.name = 'lag'
        lags = np.random.randint(1,10,1)[0]
        model = KnnEnsemble()
        model.fit(x, x, 'H', 24, lags, limit = 5, new_fit = True)
        
        self.assertEqual(False in model.X.index == model.y.index, False)
        
    def test_static(self):
        x = self.endogenous
        x = x.iloc[:,0] 
        index = x.index[int(len(x)*.85)]
        x_train = x[:index]
        x_test = x[index:]
        lags = np.random.randint(1,10,1)[0]
        model = KnnEnsemble()
        model.fit(x_train,x_train, 'H', 24, lags, limit = 5, new_fit = True)
        y_hat = model.static(x_test, test = True)
        k3 = KNeighborsRegressor(n_neighbors = 3)
        k5 = KNeighborsRegressor(n_neighbors = 5)
        k7 = KNeighborsRegressor(n_neighbors = 7)
        results = []
        x_trn = model.X
        y_trn = model.y
        for m in [k3, k5, k7]:
            m.fit(x_trn, y_trn)
            results.append(m.predict(model.x_test))
        results = np.mean(results, axis =0)
        res = results == y_hat
        self.assertEqual(False in res, False)
        
    def test_forward_selection(self):
        x = self.endogenous
        x = x.iloc[:,0] 
        index = x.index[int(len(x)*.85)]
        x_train = x[:index]
        x_test = x[index:]
        lags = 15
        model = KnnEnsemble()
        errors = model.forward_selection(x_train,x_train, x_test,x_test,'H', 24, max_lags=lags, limit = 5, brk_at_min = True)
        errors.mean()
        min_test = errors.mean()[-1]>errors.mean()[-2] and errors.mean()[-3]>errors.mean()[-2]
        self.assertEqual(min_test, True)
        

    def test_backward_selection(self):
        x = self.endogenous
        x = x.iloc[:,0] 
        index = x.index[int(len(x)*.85)]
        x_train = x[:index]
        y_train = x_train
        x_test = x[index:]
        y_test = x_test
        lags = 15
        model = KnnEnsemble()
        errors=model.backward_selection(x_train,y_train, x_test,y_test,'H', 24, lags=lags, limit = 5, brk_at_min = True)
        
    def test_forward_backward_selection(self):
        x = self.endogenous
        x = x.iloc[:,0] 
        index = x.index[int(len(x)*.85)]
        x_train = x[:index]
        y_train = x_train
        x_test = x[index:]
        y_test = x_test
        max_lags = 15
        model = KnnEnsemble()
        forward_errors = model.forward_selection(x_train,y_train, x_test,y_test,'H', 24, max_lags=max_lags, limit = 5, brk_at_min = True)
        lags = int(forward_errors.mean().idxmin().split('_')[-1]) 
        backward_errors=model.forward_backward_selection(x_train,y_train, x_test,y_test,'H', 24, max_lags=max_lags, limit = 5, brk_at_min = True)
        min_lag = int(backward_errors.mean().idxmin().split('_')[-1])
        self.assertEqual(model.X.shape[1] == lags-min_lag+1, True)
        
    def test_automatic(self):
        x = self.endogenous
        x = x.iloc[:,0] 
        y = x
        max_lags = 15
        model = KnnEnsemble()
        gr = model.automatic(x,y,'H', 24, max_lags=max_lags, limit = 5, brk_at_min = True)
        self.assertEqual(isinstance(gr, pd.core.groupby.groupby.SeriesGroupBy), True)
        
if __name__ == '__main__':
    unittest.main()
    
    
