import torch
import torch.nn as nn

class OutlierObserver(nn.Module):
    def __init__(self, threshold=3):
        super().__init__()
        self.num_total = 0
        self.mu = 0
        self.threshold = threshold

    def forward(self, x):
        print(id(self))
        if len(x.shape) == 3:
            cur_mu = torch.mean(x.detach().cpu().squeeze(0), dim=0)
            self.mu = (self.mu * self.num_total + cur_mu * x.shape[1]) / (self.num_total + 1)
            self.num_total += 1
        else:
            cur_mu = torch.mean(x.detach().cpu(), dim=0)
            self.mu = (self.mu * self.num_total + cur_mu * x.shape[0]) / (self.num_total + 1)
            self.num_total += 1

    def set_outlier_axis(self):
        #torch.save(self.mu, f"{id(self)}.pt")
        total_mu  = torch.mean(self.mu)
        total_std = torch.sqrt(torch.mean(self.mu ** 2, dtype=torch.float32) - torch.mean(self.mu, dtype=torch.float32)**2)

        z_score = (self.mu - total_mu) / total_std
        self.outlier_axis = torch.nonzero(z_score >= self.threshold)

        print("[DEBUG] z_score shape: ", z_score.shape)
        print("[DEBUG] outlier_axis: ", self.outlier_axis, " length: ", len(self.outlier_axis))


if __name__ == "__main__":
    observer = OutlierObserver()

    observer.set_outlier_axis()