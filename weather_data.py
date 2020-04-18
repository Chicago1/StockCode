import numpy as np
import requests
import pandas as pd
from pandas import DataFrame as df
import json
from datetime import datetime

# NOAA Token:
TOKEN = "NGMrBJzWTMzUoKQFuBwePKKPyVPcPjNr"

# Strings for assembling a NOAA API request:
BASE = "https://www.ncdc.noaa.gov/cdo-web/api/v2/"
ENDPOINT = "data?"
DATASET = "datasetid=GHCND&" # 'Daily Summaries'
# average temp, precipitation, cloudiness, 
DATATYPE = "datatypeid=TMAX&datatypeid=TMIN&datatypeid=PRCP&"
UNITS = "units=metric&"
LIMIT = "limit=1000&"

# Setup years:
start = 2013
end = 2020

# Setup locations:
# NYC Central Park, 
# LA Glendale,
# Chicago Midway Airport
# Houston Intercontinental Airport
# Phoenix Scottsdale Municipal Airport
stations = ["GHCND:USW00094728",
            "GHCND:USW00093134",
            "GHCND:USW00014819",
            "GHCND:USW00012960",
            "GHCND:USW00003192"]

header = {'token':TOKEN}

weather_data = {}
for station in stations:
    station_q = ("stationid="+station+"&")
    for year in range(start, end + 1):
        # We barely have enough room for ~10 months.
        # Do years in half-year increments:
        year_start = (str(year) + "-01" + "-01&")
        year_mid = (str(year) + "-06" + "-15&")
        year_end = (str(year) + "-12" + "-31&")
        for half in range(2):
            print(station, year, "H%d" % half)
            if half == 0:
                startdate = ("startdate=" + year_start)
                enddate = ("enddate=" + year_mid)
            else:
                startdate = ("startdate=" + year_mid)
                enddate = ("enddate=" + year_end)
                
            query = (BASE + ENDPOINT + DATASET + DATATYPE + station_q + 
                   LIMIT + UNITS + startdate + enddate)        
            
            r = requests.get(query, headers=header)
            d = json.loads(r.text)
            
            if d.get('results') is None:
                print("No Data.")
            else:
                for item in d['results']:
                    date_here = datetime.strptime(item['date'], "%Y-%m-%dT%H:%M:%S").strftime("%Y-%m-%d")
                    if weather_data.get(date_here) is None:
                        weather_data[date_here] = {}
                    weather_data[date_here][(item['station'] + "-" + item['datatype'])] = item['value']
           
weather_df = df.from_dict(weather_data,orient='index')
weather_df.to_csv("weather.csv",index=True, header=True)        