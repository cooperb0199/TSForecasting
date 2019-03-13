# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


from alpha_vantage_api.av import AV


class TSDF_Factory:

    
    def __init__(self, stockTicker, end="2018-12-14"):
        self.end = end
        self.ticker = stockTicker
        av = AV()
        av.pull_data(self.ticker)
    
    def createTSDF(self):
        data = pd.read_csv(f"../repositories/av_repo/{self.ticker}DailyStats.csv")
        data.date = pd.to_datetime(data.date, format = '%Y-%m-%d')
        data = data.query('date <= @self.end')
        data.index = data.date
        data = data.resample('D').mean()
        data = data.interpolate(method='linear')
        data.fillna(data.mean())
        self.data = data
        self.create_tslog()
        return data

    def split_ts(self, ts, ratio=0.80):
        """Creates a training and test set on data"""
        size = int(len(ts) * ratio)
        train, test = ts[0:size], ts[size:len(ts)]
        train.dropna(inplace=True)
        test.dropna(inplace=True)
        return train, test

    def create_tslog(self):
        self.ts_log = np.log(self.data['close'])
        self.ts_log_diff = self.ts_log - self.ts_log.shift()
        self.ts_log_diff = self.ts_log_diff.fillna(self.ts_log_diff.mean())
        self.ts_log_diff = self.ts_log_diff.mask(self.ts_log_diff == 0, self.ts_log_diff.mean())
        
        self.ts_log = self.ts_log.fillna(self.ts_log.mean())
        self.ts_log = self.ts_log.mask(self.ts_log == 0, self.ts_log.mean())

        
tsFact = TSDF_Factory('UPS')
tsFact.createTSDF()
print(tsFact.data.tail())