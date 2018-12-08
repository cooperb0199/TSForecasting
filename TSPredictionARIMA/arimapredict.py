# -*- coding: utf-8 -*-
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tsfactory import TSDF_Factory


class ArimaPredict:
    rootdir = 'Visualizations'
    
    def __init__(self, ts):
        self.ts = ts
        self.ts_log = np.log(self.ts)
        self.ts_log_diff = self.ts_log - self.ts_log.shift()
        self.ts_log_diff.dropna(inplace=True)
        
    def ar(self):
        model = ARIMA(self.ts_log, order=(1, 0, 0))  
        results_AR = model.fit(disp=-1)  
        plt.plot(self.ts_log_diff)
        plt.plot(results_AR.fittedvalues, color='red')
        plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-self.ts_log_diff)**2))
        plt.savefig("Visualizations/ar.png")
        plt.clf()
        
    def ma(self):
        model = ARIMA(self.ts_log, order=(0, 1, 2))  
        results_MA = model.fit(disp=-1)  
        plt.plot(self.ts_log_diff)
        plt.plot(results_MA.fittedvalues, color='red')
        plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-self.ts_log_diff)**2))
        plt.savefig("Visualizations/ma.png")
        plt.clf()
        
    def cm(self):
        model = ARIMA(self.ts_log, order=(2, 1, 2))  
        results_ARIMA = model.fit(disp=-1)  
        plt.plot(self.ts_log_diff)
        plt.plot(results_ARIMA.fittedvalues, color='red')
        plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-self.ts_log_diff)**2))
        plt.savefig("Visualizations/cm.png")
        plt.clf()
        predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
        print(predictions_ARIMA_diff.head())
        predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
        print(predictions_ARIMA_diff_cumsum.head())
        predictions_ARIMA_log = pd.Series(self.ts_log.ix[0], index=self.ts_log.index)
        predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
        print(predictions_ARIMA_log.head())
        
        predictions_ARIMA = np.exp(predictions_ARIMA_log)
        plt.plot(self.ts)
        plt.plot(predictions_ARIMA)
        plt.title('RMSE: %.4f'% np.sqrt(sum((predictions_ARIMA-self.ts)**2)/len(self.ts)))
        plt.savefig("Visualizations/predictions.png")
        plt.clf()
      
    @staticmethod
    def makeprediction(train, test, filename):
        history = [x for x in train]
        dateRange = pd.date_range('2016-01-14', periods=1042, freq='D')
        predictions = pd.Series('close', index=dateRange)
        f = open(f"Visualizations/{filename}_prediction.txt", "w+")
        for t in range(len(test)):
            	model = ARIMA(history, order=(0,0,1))
            	model_fit = model.fit(disp=0)
            	output = model_fit.forecast()
            	yhat = output[0]
            	predictions[t] = yhat
            	obs = test[t]
            	history.append(obs)
            	print('predicted=%f, expected=%f\n' % (yhat, obs))
            	f.write('predicted=%f, expected=%f\n' % (yhat, obs))

        error = mean_squared_error(test, predictions)
        rmse = sqrt(error)
        testArray = np.array(test.astype(np.int))
        predArray = np.round(predictions.astype(np.int))
        accuracy = accuracy_score(testArray, predArray)
        print('Test MSE: %.3f' % error)
        f.write('Test MSE: %.3f' % error)
        print('Test RMSE: %.3f' % rmse)
        f.write('Test RMSE: %.3f' % rmse)
        f.close()
        # plot
        plt.plot(test, color='black')
        plt.plot(predictions, color='red')
        plt.savefig(f'Visualizations/{filename}predictvsactual.png')
        plt.clf()