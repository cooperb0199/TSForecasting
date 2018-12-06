# -*- coding: utf-8 -*-
import pandas as pd
from tsfactory import TSDF_Factory
from arima import Arima
from trend import Trend
from arimapredict import ArimaPredict
import matplotlib.pyplot as plt


factory = TSDF_Factory()
data = factory.createTSDF()
#train, test = factory.createTSTT()
#ctrain = train['close']
#ctest = test['close']
#ctrain = ctrain.interpolate(method='linear')
#ctest = ctest.interpolate(method='linear')
#ArimaPredict.makeprediction(ctrain, ctest,'ses_woconstant')
#trend = Trend(ctrain)
#trend.findac(ctrain)

plt.plot(data['close'])
plt.savefig('Visualizations/google_close.png')


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