# -*- coding: utf-8 -*-
import pandas as pd


def get_values(data, freqstr, s, w = None, x = False, interpolate = False, limit = None, fill = False, forecast = True):
    
        
    data = data.asfreq(freqstr)
    name = data.name
    if w: 
        m = s*w
    else: 
        m = s
    if x:
        index = data[m:].index
    else:
        index = data[:-m].index
    columns = [name +'_t_'+ str(i) for i in range(1,m+1)]
    vals = pd.DataFrame(index = index, columns = columns, dtype = float) 
    vals = vals.asfreq(freqstr)
    for i in vals.index:
        if x:
            if forecast: 
                step = m + 1
            else:
                step = m
            time_stamp = i - step
        else:
            if forecast: 
                time_stamp = i+1
            else:
                time_stamp = i 
        idx = pd.date_range(time_stamp, periods=m, freq=freqstr)
        vector = data.loc[idx]
        if interpolate:
            vector = vector.interpolate(limit = limit)
        if fill:
            vector.fillna(value = vector.mean(), inplace = True)
        vals.loc[i] = vector.values
    return vals

def df_get_values(df, freqstr, s, w = None, x = False, interpolate = False, limit = None, fill = False, forecast = False):
    df_list = []
    for column in df.columns:
        data = df[column]
        d= get_values(data, freqstr, s, w = w, x = x, interpolate = interpolate, limit = limit, fill = fill, forecast = forecast)
        df_list.append(d)
    df = pd.concat(df_list, axis = 1)
    return df

