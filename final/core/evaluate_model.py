import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates 
from numpy import linalg
from core.model import Model
from os import path
from numpy import newaxis
import pickle

FILENAME = "old.npy"
SEQUENCE = 10

def mse(pred, truth):
    return(np.power(linalg.norm(pred - truth), 2)/pred.shape[0])

def sq_hinge(pred, truth, threshold):
    return(np.mean(np.power(np.maximum(0, threshold - (pred*truth)), 2)))

def average_guess_accuracy(pred, truth):
    return(np.mean(np.equal(np.sign(pred), np.sign(truth))))

# If we bet $1 each way, what happens to our money?    
def average_return_rate(pred, truth):
    return(np.mean(np.multiply(np.sign(pred),truth)))

# If we bet $1xPredictedChange each way, what happens to our money?  
def average_weighted_return_rate(pred, truth):
    principal = np.sum(np.abs(pred))
    return(np.sum(np.multiply(pred, truth))/principal)

def predict_point_by_point(predict, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    print('[Model] Predicting Point-by-Point...')
    predicted = predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted
    
def predict_sequences(predict, data, window_size, prediction_len):
    print('[Model] Predicting Sequences...')
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]            
            curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

def predict_full(predict, data, window_size):
    print('[Model] Predicting Sequences Full...')
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
    return predicted
        
# Send this function a predict() function that takes data arrays
def print_performance(predict, x_test, truth):

    #################
    ### MSE TESTS ###
    #################
    
    truth = truth.flatten()
    
    # Day-by-day MSE:
    dbd_pred = predict_point_by_point(predict, x_test)   
    dbd_mse = mse(dbd_pred, truth)
    
    # 10-Day Sequence + 10th-Day MSE:
    ten_day_seqs = np.asarray(predict_sequences(predict, x_test, 10, 10))
    
    # 10-Day Sequences:
    sequence_pred = ten_day_seqs.flatten()[:len(truth)]
    sequence_truth = truth[:len(sequence_pred)]
    sequence_mse = mse(sequence_pred, sequence_truth)
    
    # 10th-Day MSE:
    tenth_day_pred = ten_day_seqs[:-1,-1]
    tenth_day_truth = (truth[8::10])[:len(tenth_day_pred)]
    tenth_day_mse = mse(tenth_day_pred, tenth_day_truth)

    # Full-test-sequence MSE
    full_pred = np.asarray(predict_full(predict, x_test, 10))        
    full_mse = mse(full_pred, truth)
    
    print("Day-by-day MSE: %f" % dbd_mse)
    print("10-Day Sequence MSE: %f" % sequence_mse)
    print("10th-Day Sequence MSE: %f" % tenth_day_mse)
    print("Full-Prediction MSE: %f" % full_mse)    
    
    ###################
    ### OTHER TESTS ###
    ###################
    
    # Convert to day-over-day derivative:
    dbd_pred_d = np.diff(dbd_pred)/truth[:-1]
    truth_d = np.diff(truth)/truth[:-1]
    
    # Binary Loss (Percentage):
    bin_rate = average_guess_accuracy(dbd_pred_d, truth_d)
    
    # Avg. Return Rate:
    ret_rate = average_return_rate(dbd_pred_d, truth_d)
    
    # Avg. Weighted Return Rate:
    wt_ret_rate = average_weighted_return_rate(dbd_pred_d, truth_d)
    
    print("Average Binary Guess Accuracy: %f" % bin_rate)
    print("Average Simple Return: %f" % ret_rate)
    print("Average Weighted Return Rate: %f" % wt_ret_rate)
    
def plot_learning_curve(history):

    fig = plt.figure(facecolor='white')
    
    # Save off data
    np.save('loss.npy', history['loss'])  
    np.save('val_loss.npy', history['val_loss'])
    
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])        
    
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.show()    