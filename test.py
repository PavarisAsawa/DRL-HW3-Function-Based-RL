import numpy as np
import torch
import os
import torch.nn as nn


ken = np.array([[1.0, 2.0, 3.0,8.0],[1,0,0,0],[0,1,0,0],[7,6,1,3]])
tensor = torch.tensor([[1.0, 2.0, 3.0 ,4.0]])
dict = {'policy' : tensor}

# print(dict['policy'].detach().numpy())
# print(dict['policy'].detach())
print(torch.tensor(dict))
# print(dict['policy'][0])
# print(dict['policy'])

