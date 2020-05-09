import torch
import config as c
def l2_fit(input, target):
    return torch.sum((input - target)**2) / c.batch_size
