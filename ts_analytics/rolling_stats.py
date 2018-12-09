# -*- coding: utf-8 -*-
import os

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10, 6

from ts_arima.tsfactory import TSDF_Factory

class RollingStats:
    
    def find_rolling(self, ts):
        rolmean = ts.rolling(window=365).mean()
        rolstd = ts.rolling(window=365).std()
        plt.clf()
        orig = plt.plot(ts, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Mean')
        std = plt.plot(rolstd, color='black', label='Standard Deviation')
#        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        if os.path.isdir('ts_analytics/Visualizations') == False:
            os.makedirs('ts_analytics/Visualizations')
        plt.savefig('../ts_analytics/Visualizations/rolling.png', block=False)
        plt.clf()
        
    def get_cwd(self):
        return os.getcwd()
        
factory = TSDF_Factory('GOOGL')
ts = factory.createTSDF()
roll = RollingStats()
rollDir = roll.get_cwd()
roll.find_rolling(ts['close'])