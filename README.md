# TSForecasting
Forecasting Time Series Data

Procedures for running application:
1. retrieve an API key for alpha vantage from the following website: https://www.alphavantage.co
2. create a folder within the root directory called "miscellaneous" and create a text file called "api_keys.txt" within that directory.
3. Within the api_keys.txt file create a line like this: av:yourkeyhere

TSPredictionARIMA - contains classes associated with Creation of ARIMA model.
Also contains classes dealing with the assessment of the time series data.
Specifically whether the data is stationary or not.

TSPredictionARIMA/Initializer.py - Runs several standard arima models with visualizations. Also creates a visualization of the time series dataset.
