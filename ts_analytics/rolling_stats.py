# -*- coding: utf-8 -*-
import os

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6


class RollingStats:
    def __init__(self, ticker):
        self.ticker = ticker
    
    def find_rolling(self, ts, name=''):
        self.rolmean = ts.rolling(window=365).mean()
        self.rolstd = ts.rolling(window=365).std()
        plt.clf()
        orig = plt.plot(ts, color='blue', label='Original')
        mean = plt.plot(self.rolmean, color='red', label='Mean')
        std = plt.plot(self.rolstd, color='black', label='Standard Deviation')
#        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        if os.path.isdir('ts_analytics/Visualizations') == False:
            os.makedirs('ts_analytics/Visualizations')
        if os.path.isdir(f'ts_analytics/Visualizations/{self.ticker}') == False:
            os.makedirs(f'ts_analytics/Visualizations/{self.ticker}')
        plt.savefig(f'ts_analytics/Visualizations/{self.ticker}/rolling{name}.png', block=False)
        plt.clf()
        
    def get_cwd(self):
        return os.getcwd()