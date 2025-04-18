import numpy as np
import matplotlib.pyplot as plt
import os

def plot_values(data, title="Value Plot", xlabel="Index", ylabel="Value"):
    """
    Plots values from a NumPy array.
    
    Parameters:
    - data (np.ndarray): The array of values to plot.
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input data must be a NumPy array.")

    plt.figure(figsize=(8, 4))
    plt.plot(data, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def get_npy(path, file=None):
    """
    Loads .npy data from either:
    1. A full file path (if only 'path' is provided)
    2. A directory and file name (if both 'path' and 'file' are provided)
    """
    if file is None:
        fullpath = path  # Assume 'path' is full file path
    else:
        fullpath = os.path.join("experiments", path, file)

    data = np.load(fullpath, allow_pickle=True)
    # data = np.array([0 if x is None else x for x in data])
    return data

algorithm = "PPO"
exp_name = "PPO_05"
path = f"{algorithm}/{exp_name}"

reward = get_npy("experiments/Linear_Q/Linear_Q_0/duration.npy")
print(reward)