# -*- coding: utf-8 -*-
import pandas as pd

class TSDF_Factory:
    
    def createTSDF(self):
        data = pd.read_csv("../AlphaVantage/GOOGLDailyStats.csv")
        data.date = pd.to_datetime(data.date, format = '%Y-%m-%d')
        data.index = data.date
        data = data.resample('D').mean()
        return data
    
    # Creates a training and test set on data
    def createTSTT(self):
        data = self.createTSDF()
#        X = data.values
        size = int(len(data) * 0.80)
        train, test = data[0:size], data[size:len(data)]
        return train, test
