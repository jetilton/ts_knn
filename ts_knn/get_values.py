# -*- coding: utf-8 -*-
import pandas as pd



def get_values(data, freq, s, w = None, x = False, interpolate = False, limit = None, fill = False):
    
    if w: 
        m = s*w
    else: 
        m = s
    if x:
        index = data[m:].index
    else:
        index = data[:-m].index
    columns = ['t_'+ str(i) for i in range(1,m+1)]
    vals = pd.DataFrame(index = index, columns = columns, dtype = float) 
    vals = vals.asfreq(freq)
    for i in vals.index:
        if x:
            time_stamp = i - m
        else:
            time_stamp = i + 1
        idx = pd.date_range(time_stamp, periods=m, freq=freq)
        vector = data.loc[idx]
        if interpolate:
            vector = vector.interpolate(limit = limit)
        if fill:
            vector.fillna(value = vector.mean(), inplace = True)
        vals.loc[i] = vector.iloc[:,0].values
    return vals
