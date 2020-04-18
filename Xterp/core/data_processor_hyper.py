import math
import numpy as np
import pandas as pd
import time

class DataLoader():
    """A class for loading and transforming data for the lstm model

    Inputs: filename: Name of csv file with Stock and Google Trends data
            split: from CONFIG files. Sets how to partition traning and test data
            cols: from CONFIG files. Sets which columns of the csv dataset to use

    Outputs: Used as a object in main file
    """

    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)  #reads the data csv

        i_split_train = int(len(dataframe) * (split-0.15))
        print(i_split_train)
        i_split_valid = int(len(dataframe) * (0.15)) + i_split_train #sets how to split data between training and test


        self.data_train = dataframe.get(cols).values[:i_split_train] #creates train dataset from file (beginning to i_split)

        self.data_valid = dataframe.get(cols).values[i_split_train:i_split_valid]

        self.data_test  = dataframe.get(cols).values[i_split_valid:] #creates test dataset from file (i_split to end)
                                                                #this is what gets plotted at the end



        self.len_train  = len(self.data_train) #finds relavent data lengths

        self.len_valid  = len(self.data_valid) #finds relavent data lengths

        self.len_test   = len(self.data_test)


        self.len_train_windows = None #TODO: Figure out what this does

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):

            print('get test i',i)


            data_windows.append(self.data_test[i:i+seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [0]]
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''


        data_x = []
        data_y = []

        for i in range(self.len_train - seq_len):


            x, y = self._next_window_train(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)


    def get_valid_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []


        for i in range(self.len_valid- seq_len):


            x, y = self._next_window_valid(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)






    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window_train(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]

        return x, y

    def _next_window_valid(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_valid[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        y = window[-1, [0]]
        return x, y


    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i]+0.0001)) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)