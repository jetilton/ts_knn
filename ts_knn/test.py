# -*- coding: utf-8 -*-

import pandas as pd
from knn_ensemble import KnnEnsemble
import unittest
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from cwms_read.cwms_read import get_cwms


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
    
    def test_static(self):
        model = KnnEnsemble([3,5,7])
        model.fit(x_train, y_train)
        y_hat = model.static(x_test)
        k3 = KNeighborsRegressor(n_neighbors = 3)
        k5 = KNeighborsRegressor(n_neighbors = 5)
        k7 = KNeighborsRegressor(n_neighbors = 7)
        results = []
        for model in [k3, k5, k7]:
            model.fit(x_train, y_train)
            results.append(model.predict(x_test))
        results = np.mean(results, axis =0)
        res = results == y_hat
        self.assertEqual(False in res, False)
    
    def test_rmse(self):
        model = KnnEnsemble()
        model.fit(x_train, y_train)
        y_hat = model.static(x_test)
        a = model.error(x_test, y_test)
        rmse = np.sqrt((np.subtract(y_test,y_hat)**2).mean())
        res = rmse == a
        self.assertEqual(False in res, False)
        
    
    def test_forward_selection_static(self):
        model = KnnEnsemble()
        model.fit(x_train, y_train)
        results = model.forward_selection(x_train,x_test,y_train,y_test)
        rmse = results[0]
        df = results[1]
        min_index = rmse['rmse'].values.argmin()
        step = min_index + 1
        columns = x_train.columns
        cols = [x for x in columns if step >= int(x.split('_')[-1])]
        res = x_train[cols] == df
        self.assertEqual(False in res, False)
        
    def test_forward_backward_selection_static(self):
        model = KnnEnsemble()
        model.fit(x_train, y_train)
        results_f, results_fb,error, df = model.forward_backward_selection(x_train,x_test,y_train,y_test)
        if not results_f['rmse'].min() == results_fb['rmse'][0]:
            return False
        step = results_fb['step'].iloc[results_fb.index[-1] -1]
        columns = x_train.columns
        cols = [x for x in columns if step >= int(x.split('_')[-1])]
        res = x_train[cols] == df
        self.assertEqual(False in res, False)
        
        
    def test_auto(self):
        start_date = (2007, 3, 1)
        end_date =  (2010, 4,27)
        end_date_exog =  (2011, 4,27)
        endogenous = get_cwms('TDA.%-Saturation-TDG.Inst.1Hour.0.GOES-COMPUTED-REV', start_date = start_date, end_date = end_date, public = True, fill = True)
        exogenous =  get_cwms('JHAW.%-Saturation-TDG.Inst.1Hour.0.GOES-COMPUTED-REV', start_date = start_date, end_date = end_date_exog, public = True, fill = True)
        model = KnnEnsemble()
        error_list = model.automatic(endogenous=endogenous, exogenous=exogenous, freqstr = 'H')
        res = len(endo.index) != len(endogenous.index)
        self.assertEqual(res, True)
        
if __name__ == '__main__':
    unittest.main()
    
    
    
