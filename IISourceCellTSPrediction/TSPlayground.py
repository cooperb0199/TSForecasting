# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from datetime import datetime
from pandas import Series
import warnings
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
from sklearn.metrics import mean_squared_error
from math import sqrt




data = pd.read_csv("../AlphaVantage/GOOGLDailyStats.csv")

# Dimensions of dataset
n = data.shape[0]
p = data.shape[1]

#data = data.values

train_start = 0
train_end = int(np.floor(0.8*n))
test_start = train_end + 1
test_end = n
train = data[train_start:train_end]
test = data[test_start:test_end]

train_original = train.copy()
test_original = test.copy()

print(train.columns, test.columns)

train['date'] = pd.to_datetime(train.date, format = '%Y-%m-%d %H:%M')
test['date'] = pd.to_datetime(test.date, format = '%Y-%m-%d %H:%M')
train_original['date'] = pd.to_datetime(train_original.date, format = '%Y-%m-%d %H:%M')
test_original['date'] = pd.to_datetime(test_original.date, format = '%Y-%m-%d %H:%M')

for i in (train, test, train_original, test_original):
    i['year'] = i.date.dt.year
    i['month'] = i.date.dt.month
    i['day']= i.date.dt.day
    i['Hour']=i.date.dt.hour
    
train['Day of week'] = train['date'].dt.dayofweek
temp = train['date']

def applyer(row):
    if row.dayofweek == 5 or row.dayofweek == 6:
        return 1
    else:
        return 0
    
temp2 = train['date'].apply(applyer)
train['weekend'] = temp2

train.index = train['date']

ts = train['close']
plt.plot(ts)
plt.title("Time Series")
plt.xlabel("Time (year-month)")
plt.ylabel("Stock Price")
plt.legend(loc = 'best')
plt.savefig('foo.png')
plt.clf()

mode = 'month'
theMean = train.groupby(mode)['close'].mean()
uniqueMonths = train[mode].unique()
uniqueMonths.sort()
index = np.arange(len(train[mode].unique()))
plt.bar(index, theMean)
plt.xlabel(f'{mode}s', fontsize=12)
plt.ylabel('Stock Value', fontsize=12)
plt.xticks(index, uniqueMonths, fontsize=12, rotation=30)
plt.title(f'Avg Stock Price for each {mode}')
plt.savefig(f'{mode}.png')
plt.clf()


mode = 'year'
theMean = train.groupby(mode)['close'].mean()
uniqueMonths = train[mode].unique()
uniqueMonths.sort()
index = np.arange(len(train[mode].unique()))
plt.bar(index, theMean)
plt.xlabel(f'{mode}s', fontsize=12)
plt.ylabel('Stock Value', fontsize=12)
plt.xticks(index, uniqueMonths, fontsize=12, rotation=30)
plt.title(f'Avg Stock Price for each {mode}')
plt.savefig(f'{mode}.png')
plt.clf()


temp = train.groupby(['year', 'month'])['close'].mean()
temp.plot(figsize =(15,5), title = "Closing Stock(Monthwise)", fontsize = 14)
plt.savefig('foo3.png')
plt.clf()


mode = 'day'
theMean = train.groupby(mode)['close'].mean()
uniqueMonths = train[mode].unique()
uniqueMonths.sort()
index = np.arange(len(train[mode].unique()))
plt.bar(index, theMean)
plt.xlabel(f'{mode}s', fontsize=12)
plt.ylabel('Stock Value', fontsize=12)
plt.xticks(index, uniqueMonths, fontsize=12, rotation=30)
plt.title(f'Avg Stock Price for each {mode}')
plt.savefig(f'{mode}.png')
plt.clf()

train.Timestamp = pd.to_datetime(train.date, format = '%d-%m-%y %H:%M')
train.index = train.Timestamp

#Hourly
hourly = train.resample('H').mean()

#Daily
daily = train.resample('D').mean()

#Weekly
weekly = train.resample('W').mean()

#Monthly
monthly = train.resample('M').mean()

fig,axs = plt.subplots(4,1)

hourly['close'].plot(figsize = (15,8), title = "Hourly", fontsize = 14, ax = axs[0])
daily['close'].plot(figsize = (15,8), title = "Daily", fontsize = 14, ax = axs[1])
weekly['close'].plot(figsize = (15,8), title = "Weekly", fontsize = 14, ax = axs[2])
monthly['close'].plot(figsize = (15,8), title = "Monthly", fontsize = 14, ax = axs[3])
plt.savefig('foo4.png')
plt.clf()



test.Timestamp = pd.to_datetime(test.date, format='%d-%m-%Y %H:%M')
test.index = test.Timestamp

#Converting to Daily mean 
test = test.resample('D').mean()

train.Timestamp = pd.to_datetime(train.date, format='%d-%m-%Y %H:%M')
train.index = train.Timestamp

#Converting to Daily mean
#train = train.resample('D').mean()

Train = train.ix[0:2000]
valid = train.ix[1999:]

Train['close'].plot(figsize = (15,8), title = 'Monthly Stock Price', fontsize = 14, label = 'Train')
valid['close'].plot(figsize = (15,8), title = 'Monthly Stock Price', fontsize =14, label = 'Valid')
plt.xlabel('Datetime')
plt.ylabel('Closing Price')
plt.legend(loc = 'best')
plt.savefig('foo5.png')
plt.clf()


############################# Naive Approach ###########################
dd = np.asarray(Train['close'])
y_hat =valid.copy()
y_hat['naive']= dd[len(dd)- 1]
plt.figure(figsize = (12,8))
plt.plot(Train.index, Train['close'],label = 'Train')
plt.plot(valid.index, valid['close'], label = 'Validation')
plt.plot(y_hat.index, y_hat['naive'],  label = 'Naive')
plt.legend(loc = 'best')
plt.title('Naive Forecast')
plt.savefig('naive.png')
plt.clf()

rmse = sqrt(mean_squared_error(valid['close'], y_hat.naive))
print(f'Naive mean square error:\t{rmse}')


##############  Moving Avg Forecast ###########################
y_hat_avg = valid.copy()
y_hat_avg['moving_average_forecast'] = Train['close'].rolling(10).mean().iloc[-1]
plt.figure(figsize = (15,5))
plt.plot(Train['close'], label = 'Train')
plt.plot(valid['close'], label = 'Validation')
plt.plot(y_hat_avg['moving_average_forecast'], label = 'Moving Average Forecast with 10 Observations')
plt.legend(loc = 'best')
plt.savefig('MA10Observances.png')
plt.clf()
y_hat_avg = valid.copy()
y_hat_avg['moving_average_forecast'] = Train['close'].rolling(20).mean().iloc[-1]
plt.figure(figsize = (15,5))
plt.plot(Train['close'], label = 'Train')
plt.plot(valid['close'], label = 'Validation')
plt.plot(y_hat_avg['moving_average_forecast'],label = 'Moving Average Forecast with 20 Observations')
plt.legend(loc = 'best')
plt.savefig('MA20Observances.png')
plt.clf()
y_hat_avg = valid.copy()
y_hat_avg['moving_average_forecast']= Train['close'].rolling(50).mean().iloc[-1]
plt.figure(figsize = (15,5))
plt.plot(Train['close'], label = 'Train')
plt.plot(valid['close'], label = 'Validation')
plt.plot(y_hat_avg['moving_average_forecast'], label = "Moving Average Forecast with 50 Observations")
plt.legend(loc = 'best')
plt.savefig('MA50Observances.png')
plt.clf()

rmse = sqrt(mean_squared_error(valid['close'], y_hat_avg['moving_average_forecast']))
print(f'Moving Average mean square error:\t{rmse}')



##############  Simple Exponential Smoothing ###########################
from statsmodels.tsa.api import ExponentialSmoothing,SimpleExpSmoothing, Holt


y_hat = valid.copy()
fit2 = SimpleExpSmoothing(np.asarray(Train['close'])).fit(smoothing_level = 0.6,optimized = False)
y_hat['SES'] = fit2.forecast(len(valid))
plt.figure(figsize =(15,8))
plt.plot(Train['close'], label = 'Train')
plt.plot(valid['close'], label = 'Validation')
plt.plot(y_hat['SES'], label = 'Simple Exponential Smoothing')
plt.legend(loc = 'best')
plt.savefig('SES.png')
plt.clf()

rmse = sqrt(mean_squared_error(valid.close, y_hat['SES']))
print(f'SES mean square error:\t{rmse}')




############################  Holt's Linear Trend Model ################  
#plt.style.use('default')
#plt.figure(figsize = (16,8))
#import statsmodels.api as sm
#sm.tsa.seasonal_decompose(Train.close).plot()
#result = sm.tsa.stattools.adfuller(train.close)
#plt.savefig('HoltSeasonal.png')
#plt.clf()


#y_hat_holt = valid.copy()
#fit1 = Holt(np.asarray(Train['close'])).fit(smoothing_level = 0.3, smoothing_slope = 0.1)
#y_hat_holt['Holt_linear'] = fit1.forecast(len(valid))
#plt.style.use('fivethirtyeight')
#plt.figure(figsize = (15,8))
#plt.plot(Train.close, label = 'Train')
#plt.plot(valid.close, label = 'Validation')
#plt.plot(y_hat_holt['Holt_linear'], label = 'Holt Linear')
#plt.legend(loc = 'best')
#plt.savefig('HoltSeasonalPrediction.png')
#plt.clf()
#
#rmse = sqrt(mean_squared_error(valid.close, y_hat_holt.Holt_linear))
#print(f'Holt mean square error:\t{rmse}')
#
#
#
#
#
#
#predict = fit1.forecast(len(test))
#test['prediction'] = predict
#
##Calculating hourly ration of count
#train_original['ratio'] = train_original['close']/train_original['close'].sum()
#
##Grouping hourly ratio
#temp = train_original.groupby(['Hour']) ['ratio'].sum()
#
##Group by to csv format
#pd.DataFrame(temp, columns= ['Hour', 'ratio']).to_csv('Groupby.csv')
#temp2 = pd.read_csv("Groupby.csv")
#temp2 =temp2.drop('Hour.1',1)
##Merge test and test_original on day, month and year
#merge = pd.merge(test, test_original, on = ('day', 'month','year'), how = 'left')
#merge['Hour'] = merge['Hour_y']
#merge = merge.drop(['year','month','day','Hour_x','date','Hour_y'], axis =1)
#
##Predicting by merging temp2 and merge
#prediction = pd.merge(merge, temp2, on = 'Hour',how = 'left')
#
##Converting the ration to original scale
#prediction['close'] = prediction['prediction'] * prediction['ratio'] * 24
##prediction['ID'] = prediction['ID_y']
#prediction.head()
#
#prediction['ID']= prediction['ID_y']
#submission = prediction.drop(['ID_x','ID_y','day','Hour','prediction','ratio'], axis =1)
#
#pd.DataFrame(submission, columns = ['ID','close']).to_csv('Holt winters.csv')


