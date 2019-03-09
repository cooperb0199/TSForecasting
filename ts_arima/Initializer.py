# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

from ts_arima.arima import Arima
from ts_arima.arimapredict import ArimaPredict
from ts_analytics.dickey_fuller import DF
from ts_analytics.rolling_stats import RollingStats
from ts_analytics.trend import Trend
from ts_arima.tsfactory import TSDF_Factory
from utils.dir_generator import DirGen

diffModels = {
 'autoregressive' : [1,0,0]
 ,'randomwalk' : [0,1,0]
 ,'first_order_autoregressive' : [1,1,0]
 ,'simple_exponential_smoothing' : [0,1,1]
}


ticker = 'UPS'



factory = TSDF_Factory(ticker)
data = factory.createTSDF()
ctrain, ctest = factory.createTSTT(data['close'])


# Insert values for empty days
ctrain = ctrain.interpolate(method='linear')
ctest = ctest.interpolate(method='linear')

# Line graph of stock price over time
plt.plot(data['close'])
DirGen.create_dir('ts_arima/Visualizations/')
plt.savefig(f'ts_arima/Visualizations/{ticker}_close.png')

# Create graph with rolling mean and standard deviation
rollingStats = RollingStats(ticker)
rollingStats.find_rolling(data['close'])

# Dickey Fuller Test performed
df = DF()
df.perf_df_test(data['close'])

# Estimating Trend
trend = Trend(data['close'], ticker)
trend.logtransform()

# Create graph with rolling mean and standard deviation
rollingStats = RollingStats(ticker)
rollingStats.find_rolling(trend.ts_log, 'log')

logMinusMovingAvg = trend.ts_log - rollingStats.rolmean
logMinusMovingAvg.dropna(inplace=True)

# Perform Dickey Fuller Test on log minus moving avg dataset
df.perf_df_test(logMinusMovingAvg)
rollingStats.find_rolling(logMinusMovingAvg, 'logMA')


# Find the exponential weighted moving avg
expweighted_avg = trend.ewma()
logMinusEWMA = trend.ts_log - expweighted_avg
logMinusEWMA.dropna(inplace=True)

# Perform Dickey Fuller Test on log minus moving avg dataset
df.perf_df_test(logMinusEWMA)
rollingStats.find_rolling(logMinusEWMA, 'logEWMA')

# Time series with a lag of 1
numToShift = 100
datasetLogShifting = trend.ts_log - trend.ts_log.shift(numToShift)
datasetLogShifting.dropna(inplace=True)
plt.plot(datasetLogShifting)
DirGen.create_dir(f'ts_analytics/Visualizations/{ticker}')
plt.savefig(f'ts_analytics/Visualizations/{ticker}/tsShift{numToShift}.png', block=False)
plt.clf()
df.perf_df_test(datasetLogShifting)
rollingStats.find_rolling(datasetLogShifting, f'Shift{numToShift}')

# Find the seasonality and trend of the dataset
ts_log_decompose = trend.decompose()
rollingStats.find_rolling(ts_log_decompose, 'decompose')

# Find the stationarity of the residuals of this dataset
rollingStats.find_rolling(trend.residual, 'residual')
df.perf_df_test(trend.residual)

# Find the auto correlative and partial autocorrelative functions
trend.pacf()

#for item in diffModels.keys():
#    Arima.makeprediction(ctrain, ctest, item, diffModels[item][0], diffModels[item][1], diffModels[item][2])

ctrain, ctest = factory.createTSTT(datasetLogShifting)
arima = Arima(datasetLogShifting, ticker)
arima.makeprediction(ctrain, ctest, 'Custom', 2,1,2)


#arimaModel = Arima(dailyClosing)
#arimaModel.create_arima(dailyClosing, 150,1,0)

#trend.pacf()
#trend.decompose()
#trend.differencing()
#trend.extrasmoothing()
#trend.logtransform()
#trend.findMovingAvg()

#dailyClosing = daily[['close']]
#dailyClosing = dailyClosing.interpolate(method='linear')

#arimaPredict = ArimaPredict(data['close'])
#arimaPredict.ar()
#arimaPredict.ma()
#arimaPredict.cm()


#print(daily.index.freq)

#arimaModel.test_stationary(data['close'])

#trend = Trend(data['close'])
#ma = trend.findMovingAvg(trend.ts)
#ma.dropna(inplace=True)
#trend.findac(ma)