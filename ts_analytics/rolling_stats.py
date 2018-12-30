# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

from utils.dir_generator import DirGen

rcParams['figure.figsize'] = 10, 6


class RollingStats:
    def __init__(self, ticker):
        self.ticker = ticker
    
    def find_rolling(self, ts, name=''):
        self.rolmean = ts.rolling(window=365).mean()
        self.rolstd = ts.rolling(window=365).std()
        plt.clf()
        plt.plot(ts, color='blue', label='Original')
        plt.plot(self.rolmean, color='red', label='Mean')
        plt.plot(self.rolstd, color='black', label='Standard Deviation')
        plt.title('Rolling Mean & Standard Deviation')
        DirGen.create_dir(f'ts_analytics/Visualizations/{self.ticker}')
        plt.savefig(f'ts_analytics/Visualizations/{self.ticker}/rolling{name}.png', block=False)
        plt.clf()
        
