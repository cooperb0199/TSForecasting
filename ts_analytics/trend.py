# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from pandas.tools.plotting import autocorrelation_plot

from ts_arima.arima import ARIMA
from utils.dir_generator import DirGen

class Trend:
    
    def __init__(self, ts, ticker):
        self.ts = ts
        self.ts_log = np.log(self.ts)
        self.ts_log_diff = self.ts_log - self.ts_log.shift()
        self.ts_log_diff.dropna(inplace=True)
        self.ticker = ticker
    
    def logtransform(self):
        plt.plot(self.ts_log)
        plt.savefig('ts_arima/Visualizations/logTransform.png')
        plt.clf()

    def findMovingAvg(self, ts):
        moving_avg = pd.Series(ts).rolling(window=128).mean()
        plt.plot(ts)
        plt.plot(moving_avg, color='red')
        plt.savefig('ts_arima/Visualizations/movingAvg.png')
        plt.clf()
        return moving_avg
        
    def extrasmoothing(self):
        moving_avg = pd.Series(self.ts_log).rolling(window=128).mean()
        ts_log_moving_avg_diff = self.ts_log - moving_avg
        ts_log_moving_avg_diff.head(12)
        ts_log_moving_avg_diff.dropna(inplace=True)
        ARIMA.test_stationary(self, ts_log_moving_avg_diff)
        
    def ewma(self):
        expweighted_avg = pd.Series(self.ts_log).ewm(halflife=365).mean()
        plt.plot(self.ts_log, color='blue')
        plt.plot(expweighted_avg, color='red')
        plt.savefig(f'ts_arima/Visualizations/{self.ticker}ewma.png')
        plt.clf()
        return expweighted_avg
#        ts_log_ewma_diff = self.ts_log - expwighted_avg
#        ARIMA.test_stationary(self, ts_log_ewma_diff)
        
    def differencing(self):
        plt.plot(self.ts_log_diff)
        plt.savefig('ts_arima/Visualizations/diff.png')
        plt.clf()
        self.ts_log_diff.dropna(inplace=True)
        ARIMA.test_stationary(self, self.ts_log_diff)
        
    def decompose(self):
        decomposition = seasonal_decompose(self.ts_log, freq=365)
        self.trend = decomposition.trend
        self.seasonal = decomposition.seasonal
        self.residual = decomposition.resid
        
        plt.subplot(411)
        plt.plot(self.ts_log, label='Original')
        plt.legend(loc='best')
        plt.subplot(412)
        plt.plot(self.trend, label='Trend')
        plt.legend(loc='best')
        plt.subplot(413)
        plt.plot(self.seasonal,label='Seasonality')
        plt.legend(loc='best')
        plt.subplot(414)
        plt.plot(self.residual, label='Residuals')
        plt.legend(loc='best')
        plt.tight_layout()
        DirGen.create_dir(f'ts_analytics/Visualizations/{self.ticker}')
        plt.savefig(f'ts_analytics/Visualizations/{self.ticker}/decompose.png')
        plt.clf()
        ts_log_decompose = self.residual
        ts_log_decompose.dropna(inplace=True)
        return ts_log_decompose
        
    def pacf(self):
        lag_acf = acf(self.ts_log_diff, nlags=20)
        lag_pacf = pacf(self.ts_log_diff, nlags=20, method='ols')

        #Plot ACF: 
        plt.subplot(121) 
        plt.plot(lag_acf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(self.ts_log_diff)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(self.ts_log_diff)),linestyle='--',color='gray')
        plt.title('Autocorrelation Function')
        #Plot PACF:
        plt.subplot(122)
        plt.plot(lag_pacf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(self.ts_log_diff)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(self.ts_log_diff)),linestyle='--',color='gray')
        plt.title('Partial Autocorrelation Function')
        plt.tight_layout()
        plt.savefig(f'ts_analytics/Visualizations/{self.ticker}/pacf.png')
        plt.clf()
        
    def findac(self, ts):
        autocorrelation_plot(ts)
        plt.savefig('ts_analytics/Visualizations/ac.png')
        plt.clf()
        
    def testStationarity(self, ts):
        rolstd = ts.rolling(window=128).std()
        orig = plt.plot(ts,color='blue',label='Original')
        mean = plt.plot(self.findMovingAvg(ts),color='red',label='Rolling Mean')
        std = plt.plot(rolstd,color='black',label='Rolling std')
        plt.legend(loc='best')
        plt.title('Rolling mean & Standard Deviation')
        plt.show(block=False)
        

