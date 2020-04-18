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

# def add_data_to_csv(filename, data):
#     if path.exists(filename):
#         old_data = pd.read_csv(filename)
#         data = add_data_to_dataframe(old_data, data)
#     return data.to_csv(filename, index=True, header=True)
#
# # Simple - does not handle repeated columns cleverly
# def merge_csvs(filenames, output_filename):
#     data = None
#     for filename in filenames:
#         add_data_to_dataframe(data, pd.read_csv(filename))
#     return data.to_csv(output_filename, iindex=True, header=True)
    
def get_data(days, do_stocks, ticker, do_trends, words, filename):

    today = datetime.datetime.now()
    start = today - datetime.timedelta(days=days)
    end = today - datetime.timedelta(days=1)

    data = None
    stock_data = add_data_to_dataframe(data, get_stock_data(ticker, start, end))
    get_trends_data(stock_data, words, ticker, start, end, filename)

def get_stock_data(ticker, start, end):
    data = yf.download(ticker, start=str(start.date()), end=str(end.date()))
    print(data)
    data = data.drop(columns=['Adj Close'])

    export_csv_2 = data.to_csv(folder + ticker + '.csv', index=True, header=True)

    return data
 
def get_trends_data(stock_data, words, ticker, start, end, folder):

    data = stock_data

    wordLength = len(words)
    wordIndex = 0

    exceptionCounter = 0

    while True:
        try:

            prx = open('../extra/proxie_list.txt', 'r')
            Lines = prx.readlines()
            proxies = []
            for line in Lines:
                proxies.append(line.strip())



            # Login to Google. Only need to run this once, the rest of requests will use the same session.
            pytrend = TrendReq(proxies=proxies)
            # pytrend.build_payload(kw_list=['Application Software'])
            # print('pytrend.related_topics()', pytrend.related_topics())
            # #print('pytrend.suggestions(dow jones)', pytrend.suggestions('dow jones'))
            # #print('pytrend.suggestions(dow jones)', str(pytrend.categories))


            while True:

                if exceptionCounter == 3:
                    wordIndex = wordIndex+1 #goes to next word
                    exceptionCounter = 0 #resets the exceptionCounter

                    if wordIndex == wordLength - 1: #this is here if the wordIndex gets bumped too far
                        break

                word = words[wordIndex]

                trend_data = dailydata.get_daily_data(word, (start.year), (start.month), (end.year),
                                                         (end.month), wait_time=SLEEP_TIME_EACH+np.random.poisson(4.0))

                try: #if not enough trends data it will fly through this
                    trend_data = trend_data.drop(columns=['isPartial','scale',word+'_monthly',word+'_unscaled'])
                    trend_data = trend_data.rename_axis('Date')
                    export_csv = trend_data.to_csv(folder+word+'.csv', index= True, header=True)

                    #fuck it, now its saving each word as a CSV


                    time.sleep(SLEEP_TIME_GROUP)  # sleep for 15 sec so not to time out Google
                    wordIndex = wordIndex+1
                except Exception as d: #jumps to next word if this fails (usually because not enough data on a word)
                    print(d)
                    wordIndex = wordIndex + 1
                    exceptionCounter = 0
                    # print('alpha'+exceptionCounter)

                if wordIndex == wordLength-1:
                    break
            break
        except Exception as e:
            print(e)
            time.sleep(60)

            if exceptionCounter == 1:
                time.sleep(np.random.poisson(10.0)) #wait a random amount of seconds on the second attempt

            if exceptionCounter == 2:
                time.sleep(np.random.poisson(10.0)+60.0)

            print('excepted')
            exceptionCounter = exceptionCounter+1
            # print('beta' + exceptionCounter)


############################
######  Example Run  #######
############################
# Config:
ticker = "VOX"
# words = ["recession"]

# file_object  = open(ticker+'_words.txt', 'r')
file_object  = open('finance_words.txt', 'r')

Lines = file_object.readlines()
words = []
for line in Lines:
    words.append(line.strip())

days = 7*365

# Run:
do_stocks=True
do_trends=True

folder = "../data/words/"
get_data(days, do_stocks, ticker, do_trends, words, folder)

