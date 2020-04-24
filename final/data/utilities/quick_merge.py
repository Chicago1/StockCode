import pandas as pd
import os

FILELIST = 'merge_list.txt'
BASE = '..\\vox_price.csv'
OUTFILE = '..\\vox_fred.csv'

curr_df = pd.read_csv(BASE)
filelist = open(FILELIST, 'r')
files = filelist.readlines()
filelist.close()

for item in files:
    if item[-1] == '\\':
        # is directory:
        for file in os.listdir('..\\' + item):
            filename = os.fsdecode(file)
            path = ('..\\' + item + filename).strip()
            print(path)
            new_data = pd.read_csv(path)
            curr_df = curr_df.merge(new_data, how='left', sort=True)
    else:
        path = ('..\\' + item).strip()
        print(path)
        new_data = pd.read_csv(path)
        curr_df = curr_df.merge(new_data, how='left', sort=True)

curr_df = curr_df.set_index("Date").replace('.', '0.0').apply(pd.to_numeric)

curr_df = curr_df.interpolate(limit_direction='both')

print(curr_df)

export_csv = curr_df.to_csv(OUTFILE, index=True, header=True)