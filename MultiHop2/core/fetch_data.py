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

############################
######  Meta Params  #######
############################
SLEEP_TIME_EACH = 5
SLEEP_TIME_GROUP = 15

# These add_data_ functions check if the destination exists, create new if not, otherwise add to existing:
def add_data_to_dataframe(df, add):
    if df is None:
        return add
    else:
        return df.join(add)

def add_data_to_csv(filename, data):
    if path.exists(filename):
        old_data = pd.read_csv(filename)
        data = add_data_to_dataframe(old_data, data)      
    return data.to_csv(filename, index=True, header=True)            

# Simple - does not handle repeated columns cleverly    
def merge_csvs(filenames, output_filename):
    data = None
    for filename in filenames:
        add_data_to_dataframe(data, pd.read_csv(filename))
    return data.to_csv(output_filename, iindex=True, header=True)
    
def get_data(days, do_stocks, ticker, do_trends, words):

    today = datetime.datetime.now()
    start = today - datetime.timedelta(days=days)
    end = today - datetime.timedelta(days=1)

    data = None
    if do_stocks:
        data = add_data_to_dataframe(data, get_stock_data(ticker, start, end))
    if do_trends:
        data = add_data_to_dataframe(data, get_trends_data(words, ticker, start, end))
         
    return data
        
def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=str(start.date()), end=str(end.date()))
    print(data)
    data = data.drop(columns=['Adj Close'])
    return data 
 
def get_trends_data(words, ticker , start, end):

    data = None
    wordLength = len(words)
    wordIndex = 0

    exceptionCounter = 0

    while True:
        try:
            # Login to Google. Only need to run this once, the rest of requests will use the same session.
            pytrend = TrendReq()
            # pytrend.build_payload(kw_list=['Application Software'])
            # print('pytrend.related_topics()', pytrend.related_topics())
            # #print('pytrend.suggestions(dow jones)', pytrend.suggestions('dow jones'))
            # #print('pytrend.suggestions(dow jones)', str(pytrend.categories))

            done = False
            while done is False:

                if exceptionCounter == 2:
                    wordIndex = wordIndex+1
                    exceptionCounter = 0

                    if wordIndex == wordLength - 1:
                        break

                word = words[wordIndex]

                trend_data = dailydata.get_daily_data(word, (start.year), (start.month), (end.year),
                                                         (end.month), wait_time=SLEEP_TIME_EACH)



                trend_data = trend_data.drop(columns=['isPartial','scale',word+'_monthly',word+'_unscaled'])
                trend_data = trend_data.rename_axis('Date')


                data = add_data_to_dataframe(data, trend_data)

                add_data_to_csv("../data/test_7.csv", data)

                time.sleep(SLEEP_TIME_GROUP) #sleep for 15 sec so not to time out Google
                wordIndex = wordIndex+1

                if wordIndex == wordLength-1:
                    break
            break
        except Exception as e:
            print(e)
            time.sleep(60)

            if exceptionCounter == 1:
                time.sleep(np.random.poisson(10.0))

            print('excepted')
            exceptionCounter = exceptionCounter+1

    return(data)

############################
######  Example Run  #######
############################
# Config:
ticker = "VOX"
# words = ["recession"]

file_object  = open(ticker+'_words.txt', 'r')
Lines = file_object.readlines()
words = []
for line in Lines:
    words.append(line.strip())

days = 7*365
do_stocks = True
do_trends = True
# Run:
data = get_data(days, do_stocks, ticker, do_trends, words)
add_data_to_csv(filename="../data/test_7.csv", data=data)