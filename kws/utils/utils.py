import random
import numpy as np
import torch


def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def exp_moving_average(data, alpha=0.1):
    ema_data = np.zeros_like(data)
    ema_data[0] = data[0]
    
    for i in range(1, data.shape[0]):
        ema_data[i] = alpha * data[i] + (1 - alpha) * ema_data[i - 1]
    
    return ema_data
