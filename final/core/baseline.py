import numpy as np
from numpy import linalg
import pandas as pd
from sklearn .linear_model import Ridge
import matplotlib.pyplot as plt

DATA = "..\\data\\vox_price.csv"
HISTORY = 10
SPLIT_TRAIN = 0.6
SPLIT_VAL = 0.2

data = pd.read_csv(DATA, index_col=0)
data = data.to_numpy()[:,0]
m = np.mean(data)
s = np.std(data)
data = (data - m)/s

def mse(test, true):
    return(np.power(linalg.norm(test - true), 2)/len(true))

def average_guess_accuracy(test, true):
    return(np.mean(np.equal(np.sign(test), np.sign(true))))

# If we bet $1 each way, what happens to our money?    
def average_return_rate(test, true):
    return(np.mean(np.multiply(np.sign(test),true)))

# If we bet $1xPredictedChange each way, what happens to our money?  
def average_weighted_return_rate(test, true):
    principal = np.sum(np.abs(test))
    return(np.sum(np.multiply(test, true))/principal)


# A slow, risky way to do this:
x = []
y = []
for i,value in enumerate(data):
    if i >= HISTORY:
        x.append(data[i - HISTORY:i])
        y.append(value)

train_i = int(SPLIT_TRAIN*len(y))
val_i = int(SPLIT_VAL*len(y)) + train_i

###
x_train = x[:train_i]
y_train = y[:train_i]

x_val = x[train_i:val_i]
y_val = y[train_i:val_i]

x_test = x[val_i:]
y_test = y[val_i:]

### MODEL ###
alpha = 0.5
clf = Ridge(alpha=alpha)
clf.fit(x_train, y_train)


def pred_sequence(x):
    # Full Sequence prediction:
    predicted = x[0]
    for i in range(len(x)):
        p = clf.predict([predicted[i:i+HISTORY]])
        predicted = np.append(predicted, p)
    predicted = predicted[len(x[0]):]
    return(np.asarray(predicted))

def pred_partial_sequence(x):
    # Partial Sequence predictions:
    windows = []
    num_windows = int(len(x)/HISTORY)
    for i in range(num_windows):
        window = x[i*HISTORY]
        for j in range(HISTORY):
            p = clf.predict([window])
            window[:-1] = window[1:]
            window[-1] = p
        windows.append(window)    
    return(np.asarray(windows)[:-1,:])

### TEST ### 
def run_tests(x, truth):
    # Day-By-Day
    pred = clf.predict(x)

    # Partial Sequences: 
    windows = pred_partial_sequence(x)
    trimmed_len = len(windows.flatten())
    true_windows = np.reshape(truth[:trimmed_len],windows.shape)

    # 10th-Day
    furthest = windows[:,-1]
    true_furthest = true_windows[:,-1]

    # 10-Day
    windows = np.asarray(windows).flatten()
    true_windows = np.asarray(true_windows).flatten()

    # Full-Sequence:
    predicted = pred_sequence(x)

    # Other Metrics:
    pred_d = np.diff(pred)/truth[:-1]
    truth_d = np.diff(truth)/truth[:-1]

    # Binary Loss (Percentage):
    bin_rate = average_guess_accuracy(pred_d, truth_d)

    # Avg. Return Rate:
    ret_rate = average_return_rate(pred_d, truth_d)

    # Avg. Weighted Return Rate:
    wt_ret_rate = average_weighted_return_rate(pred_d, truth_d)

    plt.plot(windows)
    plt.plot(true_windows)
    plt.show()

    print("Day-by-day MSE: %f" % mse(pred,truth))
    print("Ten-Day MSE: %f" % mse(windows,true_windows))
    print("Tenth-Day MSE: %f" % mse(furthest,true_furthest))
    print("Full Series MSE: %f" % mse(predicted,truth))

    print("Average Binary Guess Accuracy: %f" % bin_rate)
    print("Average Simple Return: %f" % ret_rate)
    print("Average Weighted Return Rate: %f" % wt_ret_rate)

run_tests(x_val, y_val)
run_tests(x_test, y_test)