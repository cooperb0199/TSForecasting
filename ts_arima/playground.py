# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt


from arima import Arima
from arimapredict import ArimaPredict
from alpha_vantage_api.av import AV
from trend import Trend
from tsfactory import TSDF_Factory
from ts_analytics.rolling_stats import RollingStats

av = AV()

factory = TSDF_Factory('GOOGL')
ts = factory.createTSDF()
rollStats = RollingStats()
curDir = rollStats.get_cwd()
rollStats.find_rolling(ts)
#data = factory.createTSDF('GOOGL')
#
#trend = Trend(data['close'])
#trend.testStationarity(data['close'])


