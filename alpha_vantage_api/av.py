# -*- coding: utf-8 -*-
import os

from alpha_vantage.timeseries import TimeSeries
import datetime

class AV:
    """This class pulls data from alpha vantage API
    If the data is being request within the day span
    from the last time a pull occurred, then the
    data will be pulled from a cached csv file of
    the stock info.
    *** Note: you must retrieve your own API key from
    alpha vantage, and put it into the api_keys.txt file
    within the "miscellaneous" class under your root
    directory. the format should be like:
    "av:yourkeyhere"
    """
    apiKeys = {}
    
    def __init__(self):
        keyfile = open("../miscellaneous/api_keys.txt")
        content = keyfile.readlines()
        for item in content:
            lines = item.split(":")
            self.apiKeys[lines[0]] = lines[1]
        if os.path.isdir('../repositories/av_repo') == False:
            os.makedirs('../repositories/av_repo')
    
    def pull_data(self, ticker):
        """Pull data from either csv or API
        depending on the previous pull. 
        Provide the stock ticker to pull
        data for the desired stock.
        """
        if os.path.isfile(f"../miscellaneous/lastRun{ticker}.txt") == True:
            file = open(f"../miscellaneous/lastRun{ticker}.txt", "a+")
            fileread = open(f"../miscellaneous/lastRun{ticker}.txt", "r+")
        else:
            file = open(f"../miscellaneous/lastRun{ticker}.txt", "w+")
        contents = fileread.readlines()
        currentDate = datetime.datetime.now()
        contentCount = len(contents)
        if contentCount > 0 :
            ts_string = contents[contentCount - 1].rstrip()
            timeStamp = datetime.datetime.strptime(ts_string, "%m/%d/%Y %H:%M")
            timeDelta = currentDate - timeStamp
            if (timeDelta.days >= 1):
                self.pull_from_api(ticker)
        else :
            self.pull_from_api(ticker)
        file.write(f"{currentDate.strftime('%m/%d/%Y %H:%M')}\r\n")
        

        
    def pull_from_api(self, ticker):
        """Pull stock information from API
        """
        ts = TimeSeries(key=self.apiKeys['av'], output_format='pandas')
        # Get json object with the intraday data and another with  the call's metadata
        data, meta_data = ts.get_daily('GOOGL', outputsize='full')
        data.columns = ['open','high','low','close','volume']
        data.to_csv(f'../repositories/av_repo/{ticker}DailyStats.csv', sep=',')
        
        