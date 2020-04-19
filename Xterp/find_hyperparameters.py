'''
Unused imports
'''
import sys #used to make pauses to code
import time #not sure (could use to time the program)
import math #not sure
from matplotlib.dates import DateFormatter
'''
Imports
'''
import os #used for setting that lets tensor flow
import json #used in the config files:
            #confinWithoutTrends.json and configWithTrends.json

import matplotlib.pyplot as plt #used for plotting
import pandas as pd #used for databases
import numpy as np #used for calculations

import matplotlib.dates as mdates #dates for plotting/ estimating

import datetime #datetime for configuring which dates to extract from getStocks

import time

from core.data_processor_hyper import DataLoader #no error, from the core folder
from core.model_hyper import LSTMHyperModel

from kerastuner.tuners import RandomSearch

from core.evaluate_model import predict_point_by_point
from core.evaluate_model import predict_sequences_multiple

from tensorflow.keras.models import Sequential, load_model

from tensorflow.keras.callbacks import EarlyStopping







#MAKE SURE BOTH DATASETS (yahoo stock and google trends) EXACT SAME LENGTH AND FILLED
'''
Runs main code. TODO: Make it into a function that inputs: "Ticker", "Dates of Interest", "Trendword 1", "Trendword 2", etc...

Inputs: None
Outputs: Plot with stock fluctuations as a percent change from the start of window
'''

configs = json.load(open('configWithTrends.json', 'r'))

#Creates a new DataLoader object, see core/data_processor to see what it does.
data = DataLoader(
    configs['data']['filename'],
    configs['data']['train_test_split'],
    configs['data']['columns']
)

#Creates model save directory
if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])

#Creates a new Model object, see core/model to see what it does.
# model.build_model(configs)

#TODO: Make Bayesian happen

tuner = RandomSearch(
    LSTMHyperModel(),
    objective = 'val_loss',
    max_trials = 100
)


x_train, y_train = data.get_train_data(
    seq_len=configs['data']['sequence_length'],
    normalise=configs['data']['normalise']
)

x_valid, y_valid = data.get_valid_data(
    seq_len=configs['data']['sequence_length'],
    normalise=configs['data']['normalise']
)

# tuner.search(x_train,y_train, validation_data = (x_valid,y_valid),epochs = 200)

# Show a summary of the search
tuner.results_summary()


best_hps = tuner.get_best_hyperparameters()[0]
print(best_hps.values)


