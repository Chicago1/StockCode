import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg
from core.model import Model
from os import path

FILENAME = "old.npy"

def mse(pred, truth):
    return(np.power(linalg.norm(pred - truth), 2))

def sq_hinge(pred, truth, threshold):
    return(np.sum(np.power(np.maximum(0, threshold - (pred*truth)), 2)))

def average_guess_accuracy(pred, truth):
    return(np.mean(np.equal(np.sign(pred), np.sign(truth))))

# If we bet $1 each way, what happens to our money?    
def average_return_rate(pred, truth):
    return(np.mean(np.sign(pred)*truth))
    
def print_performance(model, x_test, truth):
    pred = model.predict_point_by_point(x_test)
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