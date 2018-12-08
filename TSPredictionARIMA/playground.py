# -*- coding: utf-8 -*-

import pandas as pd
from tsfactory import TSDF_Factory
from arima import Arima
from trend import Trend
from arimapredict import ArimaPredict
import matplotlib.pyplot as plt


factory = TSDF_Factory()
data = factory.createTSDF('GOOGL')

trend = Trend(data['close'])
trend.testStationarity(data['close'])