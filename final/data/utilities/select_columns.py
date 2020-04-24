import pandas as pd
from pandas import DataFrame as df

FILENAME_IN = "..\\vox_reddit.csv"
FILENAME_OUT = "..\\vox_reddit.csv"

COLUMNS = [ "Date",
            "reddit-Advertising",
            "reddit-Alternative Carriers",
            "reddit-Broadcasting",
            "reddit-Cable",
            "reddit-Interactive Media",
            "reddit-Movies",
            ]

data = pd.read_csv(FILENAME_IN, index_col=0, usecols=COLUMNS)
data.to_csv(FILENAME_OUT,index=True, header=True)