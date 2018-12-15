# -*- coding: utf-8 -*-
from statsmodels.tsa.stattools import adfuller
import pandas as pd

from ts_arima.tsfactory import TSDF_Factory


class DF:
    def perf_df_test(self, ts):
        print('Results of Dickey-Fuller test')
        dftest = adfuller(ts, autolag='AIC')
        
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)
    