import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_value(filepath , filename):
    value_arr = np.load(file=f"{filepath}/{filename}",allow_pickle=True)
    for i,val in enumerate(value_arr):
        if val is None:
            value_arr[i] = 0
    return value_arr

def plot(value, file_name, window_size=1):
    '''
    Input:
        value       : List-like values (1D)
        file_name   : Title of the plot
        window_size : Size of the moving average window (default=1, means no smoothing)
    '''
    value = np.array(value)

    if window_size > 1:
        weights = np.ones(window_size) / window_size
        smoothed = np.convolve(value, weights, mode='valid')
        indices = range(window_size - 1, len(value))
    else:
        smoothed = value
        indices = range(len(value))

    plt.plot(indices, smoothed)
    plt.title(file_name + f' (MA{window_size})' if window_size > 1 else file_name)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True)
    plt.show()

# Algorithm_name = "MC_REINFORCE"
# experiment_name = "MC_REINFORCE_1"
Algorithm_name = "MC_REINFORCE"
experiment_name = "MC_REINFORCE_base"
fullpath = f"experiments/{Algorithm_name}/{experiment_name}"

val = load_value(filepath=fullpath , filename="duration.npy")
plot(value=val , file_name=experiment_name , window_size=100)


