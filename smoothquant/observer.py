import torch
import torch.nn as nn

class OutlierObserver(nn.Module):
    def __init__(self):
        self.num_total = 0
        self.mu_sqaure = 0.0
        self.mu = 0.0
        self.std = 0.0

    def forward(self, x):
        self.mu = (self.mu * self.num_total )
        self.num_total += x.shape[0]