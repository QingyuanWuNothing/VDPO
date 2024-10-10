import os
from itertools import product
import random
import torch
import numpy as np

def find_specific_exp_tag(exp_dict, exp_tag):
    contents = os.listdir(exp_dict)
    for item in contents:
        if exp_tag in item:
            return True
    return False

def make_seeding(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def make_anneal_lr(update, num_updates, lr):
    frac = 1.0 - (update - 1.0) / num_updates
    return frac * lr

def make_linear_schedule(start, end, duration, t):
    slope = (end - start) / duration
    return max(slope * t + start, end)
