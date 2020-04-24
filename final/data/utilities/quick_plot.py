from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np

import matplotlib.dates as mdates #dates for plotting/ estimating
import datetime #datetime for configuring which dates to extract from getStocks

import time

from statsmodels.tsa.filters.hp_filter import hpfilter

fig = plt.figure(facecolor='white',figsize = (12,5))
ax = fig.add_subplot(111)

for file in os.listdir():
     filename = os.fsdecode(file)
     if filename.endswith("X.csv")  or filename.endswith("uct.csv"):
         print(os.path.join(filename))
         new_data = pd.read_csv(os.path.join(filename))
         x_data = new_data[new_data.columns[0]]
         y_data = new_data[new_data.columns[1]]

         y_data = (y_data - min(y_data)) / (max(y_data) - min(y_data))

         time.sleep(1)

         dates = mdates.date2num(pd.to_datetime(x_data))

         months = mdates.MonthLocator()  # every month
         months_fmt = mdates.DateFormatter('%Y-%m')

         plt.plot_date(dates, y_data, fmt="-",  linewidth="0.5")



         ax.xaxis.set_major_locator(months)
         ax.xaxis.set_major_formatter(months_fmt)

         ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
         ax.grid(True)



plt.show()




