import torch
import torch.nn as nn

class OutlierObserver(nn.Module):
    def __init__(self, threshold=3):
        self.num_total = 0
        self.mu = 0.0
        self.threshold = threshold

    def forward(self, x):
        if len(x.shape) == 3:
            cur_mu = torch.mean(x.detach().cpu().squeeze(0), dim=0)
            self.mu = (self.mu * self.num_total + cur_mu * x.shape[1]) / (self.num_total + x.shape[1])
            self.num_total += x.shape[1]
        else:
            cur_mu = torch.mean(x.detach().cpu(), dim=0)
            self.mu = (self.mu * self.num_total + cur_mu * x.shape[0]) / (self.num_total + x.shape[0])
            self.num_total += x.shape[0]

    def set_outlier_axis(self):
        total_mu  = torch.mean(self.mu)
        total_std = torch.sqrt(torch.mean(self.mu ** 2) - torch.mean(self.mu)**2)

        z_score = (self.mu - total_mu) / total_std
        self.outlier_axis = torch.nonzero(z_score >= self.threshold)

        print("[DEBUG] z_score shape: ", z_score.shape)
        print("[DEBUG] outlier_axis: ", self.outlier_axis)