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

avg_stock = 0

for file in os.listdir():
     filename = os.fsdecode(file)
     if filename.endswith("Q.csv"):
         print(os.path.join(filename))
         new_data = pd.read_csv(os.path.join(filename))
         x_data = new_data[new_data.columns[0]]
         y_data = new_data[new_data.columns[1]]


         y_data = [((float(p) / float(y_data[0] + 0.0001)) - 1) for p in y_data]
         #y_data = (y_data - min(y_data)) / (max(y_data) - min(y_data))

         avg_stock = np.average(y_data)

         time.sleep(1)

         dates = mdates.date2num(pd.to_datetime(x_data))

         months = mdates.MonthLocator()  # every month
         months_fmt = mdates.DateFormatter('%Y-%m')

         plt.plot_date(dates, y_data, fmt="-", linewidth="0.5")

         ax.xaxis.set_major_locator(months)
         ax.xaxis.set_major_formatter(months_fmt)

         ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
         ax.grid(True)

for file in os.listdir():
     filename = os.fsdecode(file)
     if filename.endswith("el.csv"):
         print(os.path.join(filename))
         new_data = pd.read_csv(os.path.join(filename))
         x_data = new_data[new_data.columns[0]]
         y_data = new_data[new_data.columns[1]]

         y_data = ((y_data - min(y_data)) / (max(y_data) - min(y_data)))

         y_data = y_data-(np.average(y_data) - avg_stock)

         cycle, y_data_smooth = hpfilter(y_data, lamb=50000000)

         time.sleep(1)

         dates = mdates.date2num(pd.to_datetime(x_data))

         months = mdates.MonthLocator()  # every month
         months_fmt = mdates.DateFormatter('%Y-%m')

         # plt.plot_date(dates, y_data, fmt="-",  linewidth="0.5")
         plt.plot_date(dates, y_data_smooth, fmt="-",  linewidth="0.5")



         ax.xaxis.set_major_locator(months)
         ax.xaxis.set_major_formatter(months_fmt)

         ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
         ax.grid(True)



plt.show()




