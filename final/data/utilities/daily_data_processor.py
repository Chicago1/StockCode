import datetime
from os import path
from time import mktime
import time
import yfinance as yf
from pytrends.request import TrendReq #for the getTrends section (downloads Google trends data)
from pytrends import dailydata #trick to get long term normalized data from Google trends
import pandas as pd
import numpy as np
import sys




def cut_csv(days, filename):
    today = datetime.datetime.now() #sets the dates
    start = today - datetime.timedelta(days=days)
    end = today - datetime.timedelta(days=1)

    old_data = pd.read_csv(filename)

    # old_data = old_data.rename(columns={'DATE':''})


    row_df = pd.DataFrame([str(today)], columns={'DATE'})
    # row_df = row_df.rename(columns={'':'DATE'})
    # print(row_df)

    old_data = pd.concat([old_data, row_df ], ignore_index=True, sort=True)
    old_data['DATE'] = pd.to_datetime(old_data['DATE'])
    old_data = old_data.set_index('DATE').resample('D').interpolate()
    old_data = old_data[-days:]
    old_data = old_data.rename_axis('Date')
    old_data.to_csv('processed' + filename, index=True, header=True)
    print(old_data)


cut_csv(7*365, 'HSN1F.csv' )






