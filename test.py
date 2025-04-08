import numpy as np
import torch
import os
import torch.nn as nn

def save_episode_duration(path, filename):
    """
    Save Episode Duration parameters.
    """
    # ========= put your code here ========= #
    if not os.path.exists(path):
        os.makedirs(path)
    filepath = os.path.join(path, filename)
    np.save(filepath, [4,21,34,84,5,4,6,75,4,3123])

folder = "ken"
save_episode_duration(path=f"{folder}/asdas",filename="test")