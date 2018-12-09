# -*- coding: utf-8 -*-
import os

import pandas as pd

from alpha_vantage_api.av import AV


class TSDF_Factory:

    
    def __init__(self, stockTicker):
        self.ticker = stockTicker
        av = AV()
        av.pull_data(self.ticker)
    
    def createTSDF(self):
        data = pd.read_csv(f"../repositories/av_repo/{self.ticker}DailyStats.csv")
        data.date = pd.to_datetime(data.date, format = '%Y-%m-%d')
        data.index = data.date
        data = data.resample('D').mean()
        data = data.interpolate(method='linear')
        return data
    

    def createTSTT(self):
        """Creates a training and test set on data
        """
        data = self.createTSDF(self.ticker)
        size = int(len(data) * 0.80)
        train, test = data[0:size], data[size:len(data)]
        return train, test

