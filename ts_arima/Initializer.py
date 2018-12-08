# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

from tsfactory import TSDF_Factory
from arima import Arima
from trend import Trend
from arimapredict import ArimaPredict


diffModels = {
 'autoregressive' : [1,0,0]
 ,'randomwalk' : [0,1,0]
 ,'first_order_autoregressive' : [1,1,0]
 ,'simple_exponential_smoothing' : [0,1,1]
}


ticker = 'GOOGL'



factory = TSDF_Factory()
data = factory.createTSDF(ticker)
train, test = factory.createTSTT(ticker)
ctrain = train['close']
ctest = test['close']

# Insert values for empty days
ctrain = ctrain.interpolate(method='linear')
ctest = ctest.interpolate(method='linear')


#ArimaPredict.makeprediction(ctrain, ctest,'ses_woconstant')
trend = Trend(data['close'])
trend.testStationarity(data['close'])
#trend.findac(ctrain)

# Line graph of google stock over time
plt.plot(data['close'])
plt.savefig('Visualizations/google_close.png')

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