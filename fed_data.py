import numpy as np
import requests
import pandas as pd
from pandas import DataFrame as df
import json
from datetime import datetime

# FRED Token:
API_KEY = "adcc4a62b6081eac047cb8a4991d8905"

FILE_TYPE = "json"
ENDPOINT = "https://api.stlouisfed.org/fred/series/observations?"

# Setup Years:
start = 2013
end = 2020

# Data Series:
# BofA US High Yield Index Option-Adjusted Spread
# 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
# Daily Fed Funds Rate
# U.S. / Euro Foreign Exchange Rate
# 10-Year Breakeven Inflation Rate
# 10-Year Treasury Constant Maturity Rate
# 10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity 
# TED Spread
# 3-Month London Interbank Offered Rate (LIBOR), based on U.S. Dollar
# ICE BofA BBB US Corporate Index Option-Adjusted Spread
series_ids = ["BAMLH0A0HYM2",
              "T10Y2Y",
              "DFF",
              "DEXUSEU",
              "T10YIE",
              "DGS10",
              "T10Y3M",
              "TEDRATE",
              "USD3MTD156N",
              "BAMLC0A4CBBB"]

fred_data = {}
date_start = (str(start) + "-01" + "-01")
date_end = (str(start) + "-12" + "-31")
for series_id in series_ids:
    print(series_id)
    query = (ENDPOINT + "series_id=" + series_id + "&api_key=" + API_KEY + "&file_type=" + FILE_TYPE
             + "&observation_start=" + date_start) # Default date range end is TODAY
    r = requests.get(query)
    d = json.loads(r.text)
    
    for item in d['observations']:
        if fred_data.get(item['date']) is None:
            fred_data[item['date']] = {}
        fred_data[item['date']][series_id] = item['value']
                
fred_df = df.from_dict(fred_data,orient='index')
fred_df.to_csv("fred.csv",index=True, header=True)   