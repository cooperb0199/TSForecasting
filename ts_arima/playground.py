# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt


from arima import Arima
from arimapredict import ArimaPredict
from av import AV
from trend import Trend
from tsfactory import TSDF_Factory

av = AV()

#factory = TSDF_Factory()
#data = factory.createTSDF('GOOGL')
#
#trend = Trend(data['close'])
#trend.testStationarity(data['close'])


