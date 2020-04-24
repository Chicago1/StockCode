import pandas as pd
import os
import copy
import numpy as np

merged_data = pd.read_csv('joined.csv')

empty_data = pd.DataFrame(columns = merged_data.columns)


date_data = (merged_data.iloc[:, 0])

empty_data['Date'] = date_data[365:len(merged_data) - 1]

column_list = merged_data.columns[1:] #first columns that are not normalized stock price

i=1
for column in column_list:

    y_data = merged_data[column]

    if i > 5:
        y_data_copy = copy.deepcopy(y_data)
        y_data_copy = np.pad(y_data_copy, (365, 0), 'constant', constant_values=(1))
        y_data_old = y_data_copy[0:len(y_data_copy) - 365]
        y_data_new = []
        for i in range(len(y_data_old)):
            if y_data_old[i] == 0:
                y_data_new.append(0)
            else:
                y_data_new.append((y_data[i] - y_data_old[i]) / (y_data_old[i]))
        y_data = y_data_new[366:len(y_data_copy) - 1]

        print(len(empty_data[column]),'col')
        print(len(y_data_new),'ydat')

    empty_data[column] = y_data
    i=i+1

print(merged_data.head)
print(empty_data.head)

export_csv = empty_data.to_csv('joined_yovery.csv', index=False, header=True)




