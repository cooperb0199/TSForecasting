# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

from ts_arima.arima import Arima
from ts_arima.arimapredict import ArimaPredict
from ts_arima.trend import Trend
from ts_analytics.dickey_fuller import DF
from ts_analytics.rolling_stats import RollingStats
from ts_arima.tsfactory import TSDF_Factory


diffModels = {
 'autoregressive' : [1,0,0]
 ,'randomwalk' : [0,1,0]
 ,'first_order_autoregressive' : [1,1,0]
 ,'simple_exponential_smoothing' : [0,1,1]
}


ticker = 'GOOGL'



factory = TSDF_Factory(ticker)
data = factory.createTSDF()
train, test = factory.createTSTT()
ctrain = train['close']
ctest = test['close']

# Insert values for empty days
ctrain = ctrain.interpolate(method='linear')
ctest = ctest.interpolate(method='linear')

# Line graph of google stock over time
plt.plot(data['close'])
plt.savefig('ts_arima/Visualizations/google_close.png')

# Create graph with rolling mean and standard deviation
rollingStats = RollingStats()
rollingStats.find_rolling(data['close'])

# Dickey Fuller Test performed
df = DF()
df.perf_df_test(data['close'])

# Estimating Trend
trend = Trend(data['close'])
trend.logtransform()


for item in diffModels.keys():
    Arima.makeprediction(ctrain, ctest, item, diffModels[item][0], diffModels[item][1], diffModels[item][2])


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