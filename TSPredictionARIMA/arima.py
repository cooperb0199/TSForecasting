# -*- coding: utf-8 -*-
from statsmodels.tsa.stattools import adfuller
from matplotlib.pylab import rcParams
from tsfactory import TSDF_Factory
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame


import matplotlib.pyplot as plt
import pandas as pd


    

class Arima:
    def __init__(self, timeseries):
        self.ts = timeseries
    
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
        plt.savefig("Visualizations/ARIMA.png")
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
        model_fit = model.fit(disp=0)
        print(model_fit.summary())
        # plot residual errors
        residuals = DataFrame(model_fit.resid)
        residuals.plot()
        plt.savefig('Visualizations/residError.png')
        plt.clf()
        residuals.plot(kind='kde')
        plt.savefig('Visualizations/redensity.png')
        plt.clf()
        print(residuals.describe())
        
    @staticmethod
    def makeprediction(train, test, filename, lag, diff, ma):
        history = [x for x in train]
        dateRange = pd.date_range('2016-01-14', periods=1042, freq='D')
        predictions = pd.Series('close', index=dateRange)
        f = open(f"Visualizations/{filename}_prediction.txt", "w+")
        for t in range(len(test)):
            	model = ARIMA(history, order=(lag,diff,ma))
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