# -*- coding: utf-8 -*-
from alpha_vantage.timeseries import TimeSeries

ts = TimeSeries(key='KITJ5MYFLLIN6QE1', output_format='pandas')
# Get json object with the intraday data and another with  the call's metadata
data, meta_data = ts.get_daily('GOOGL', outputsize='full')
data.columns = ['open','high','low','close','volume']
data.to_csv('GOOGLDailyStats.csv', sep=',')


