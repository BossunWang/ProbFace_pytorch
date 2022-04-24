import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class UncertaintyHead(nn.Module):
    ''' Evaluate the log(sigma^2) '''
    
    def __init__(self, in_feat=512):

        super(UncertaintyHead, self).__init__()
        self.fc1 = nn.Linear(in_feat, 256)
        self.bn1 = nn.BatchNorm1d(256, affine=True)
        self.relu = nn.ReLU(in_feat)
        self.fc2 = nn.Linear(256, 1)
        self.bn2 = nn.BatchNorm1d(1, affine=False)
        self.gamma = Parameter(torch.Tensor([0.1]))
        self.beta = Parameter(torch.Tensor([-1.0]))   # default = -7.0

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))  # 2*log(sigma)
        x = self.gamma * x + self.beta
        x = torch.log(1e-6 + torch.exp(x))  # log(sigma^2)
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_feat = 47040
    batch_size = 2
    unh = UncertaintyHead(in_feat=47040).to(device)
    input = torch.randn(batch_size, in_feat).to(device)
    log_sigma_sq = unh(input)
    print(log_sigma_sq)
