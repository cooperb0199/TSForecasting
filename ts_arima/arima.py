# -*- coding: utf-8 -*-
from math import sqrt
from numpy.linalg import LinAlgError
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ts_arima.tsfactory import TSDF_Factory
from utils.dir_generator import DirGen
    

class Arima:
    def __init__(self, timeseries, ticker):
        self.ts = timeseries
        self.ticker = ticker
        DirGen.create_dir(f'ts_forecast/Visualizations/{ticker}/')
    
    def test_stationary(self):
        #Determine rolling statistics
        rolmean = pd.Series(self.ts).rolling(window = 200).mean()
        rolstd = pd.Series(self.ts).rolling(window = 200).std()
        
        #Plot rolling Statistics
        orig = plt.plot(self.ts, color = "blue", label = "Original")
        mean = plt.plot(rolmean, color = "red", label = "Rolling Mean")
        std = plt.plot(rolstd, color = "black", label = "Rolling Std")
        plt.legend(loc = "best")
        plt.title("Rolling Mean and Standard Deviation")
        #plt.show(block = False)
        plt.savefig("ts_arima/Visualizations/ARIMA.png")
        plt.clf()
        #Perform Dickey Fuller test
        print("Results of Dickey Fuller test: ")
        dftest = adfuller(self.ts, autolag = 'AIC')
        dfoutput = pd.Series(dftest[0:4], index = ['Test Statistics', 'p-value', '# Lag Used', 'Number of Observations Used'])
        
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)' %key] = value
        print(dfoutput)
        
    def create_arima(self, ts, p, d, q):
        #fit model
        model = ARIMA(ts, order=(p,d,q))
        model_fit = model.fit(disp=-1)
        print(model_fit.summary())
        # plot residual errors
        residuals = DataFrame(model_fit.resid)
        residuals.plot()
        plt.savefig('ts_arima/Visualizations/residError.png')
        plt.clf()
        residuals.plot(kind='kde')
        plt.savefig('ts_arima/Visualizations/redensity.png')
        plt.clf()
        print(residuals.describe())
        
    def fnd_resid_frm_arima(self, ts, p, d , q):
        model = ARIMA(ts.ts_log, order=(p,d,q))
        model_fit = model.fit(disp=-1)
        plt.plot(ts.ts_log_diff)
        plt.plot(model_fit.fittedvalues, color='red')
        tslog = ts.ts_log
#        modelfit = model_fit.fittedvalues
        num1 = model_fit.fittedvalues - ts.ts_log_diff
        num1.dropna(inplace=True)
        plt.title('RSS: %.4f'% sum(num1**2))
        plt.savefig(f'ts_forecast/Visualizations/{self.ticker}/arimaResid.png')
        modelfit_cumsum = model_fit.fittedvalues.cumsum()
        print(modelfit_cumsum.head())
        predictions_ARIMA_log = pd.Series(tslog.ix[0],index=tslog.index)
        logvaluetest = predictions_ARIMA_log #TODO debug
        predictions_ARIMA_log = predictions_ARIMA_log.add(modelfit_cumsum,fill_value=0)
        print(predictions_ARIMA_log.head())
        plt.clf()
        predictions_ARIMA = np.exp(predictions_ARIMA_log)
        plt.plot(ts.data['close'])
        plt.plot(predictions_ARIMA)
        plt.clf()
        model_fit.plot_predict(1,10625)
        plt.savefig(f'ts_forecast/Visualizations/{self.ticker}/predictionwconf.png')
        
    
    def makeprediction(self, train, test, filename, lag, diff, ma):
        history = [x for x in train]
        dateRange = pd.date_range('2015-02-19', periods=1395, freq='D')
        predictions = pd.Series('close', index=dateRange)
        index = 1
        f = open(f"ts_forecast/Visualizations/{self.ticker}/{filename}_prediction.txt", "w+")
        for t in range(len(test)):
            try:
                	model = ARIMA(history, order=(lag,diff,ma))
                	model_fit = model.fit(disp=-1)
                	output = model_fit.forecast()
                	yhat = output[0]
                	predictions[t] = yhat
                	obs = test[t]
                	history.append(obs)
                	print(f'{index}: predicted=%f, expected=%f\n' % (yhat, obs))
                	f.write('predicted=%f, expected=%f\n' % (yhat, obs))
            except:
                print(f'error with data: {history[-1]}')
                obs = test[t]
                history.append(obs)
            index += 1
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
        plt.savefig(f'ts_forecast/Visualizations/{self.ticker}/{filename}predictvsactual.png')
        plt.clf()
        


ticker = 'UPS'
factory = TSDF_Factory(ticker)
data = factory.createTSDF()      
ctrain, ctest = factory.createTSTT(factory.ts_log) 
arima = Arima(data, ticker)
#arima.makeprediction(ctrain, ctest, 'tst', 2, 1, 2)
arima.fnd_resid_frm_arima(factory,2,1,2)