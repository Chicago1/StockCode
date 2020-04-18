import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from core.model import Model
from os import path
from numpy import newaxis

FILENAME = "old.npy"

def mse(pred, truth):
    return(np.power(linalg.norm(pred - truth), 2))

def sq_hinge(pred, truth, threshold):
    return(np.mean(np.power(np.maximum(0, threshold - (pred*truth)), 2)))

def average_guess_accuracy(pred, truth):
    return(np.mean(np.equal(np.sign(pred), np.sign(truth))))

# If we bet $1 each way, what happens to our money?    
def average_return_rate(pred, truth):
    return(np.mean(np.sign(pred)*truth))

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    print('[Model] Predicting Point-by-Point...')
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted
    
def predict_sequences_multiple(model, data, window_size, prediction_len):
    print('[Model] Predicting Sequences Multiple...')
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def predict_full(model, data, window_size):
    print('[Model] Predicting Sequences Full...')
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
    return predicted

def print_performance(model, x_test, truth):
    pred = predict_point_by_point(model, x_test)
    
    truth = truth.reshape(pred.shape)
    # Produce a percent day-over-day change
    predicted_daily_change = np.diff(pred)/truth[:-1]
    actual_daily_change = np.diff(truth)/truth[:-1]
    
    mean_sq_here = mse(predicted_daily_change, actual_daily_change)
    sq_hinge_here = sq_hinge(predicted_daily_change, actual_daily_change, 1)
    avg_guess_here = 100*average_guess_accuracy(predicted_daily_change, actual_daily_change)
    avg_return_here = average_return_rate(predicted_daily_change, actual_daily_change)
    
    if path.exists(FILENAME):
        old = np.load(FILENAME)        
    else:
        old = [0.0, 0.0, 0.0, 0.0]
    
    print("MSE change prediction: %f (last: %f)" % (mean_sq_here, old[0]))
    print("Sum Sq. Hinge Loss on change prediction: %f (last: %f)" % (sq_hinge_here, old[1]))
    print("Percent chance of guessing right: %f (last: %f)" % (avg_guess_here, old[2]))
    print("Simple strategy returns: %f (last: %f)" % (avg_return_here, old[3]))
    
    np.save(FILENAME, np.asarray([mean_sq_here, sq_hinge_here, avg_guess_here, avg_return_here]))