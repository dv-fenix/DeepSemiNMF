import torch
import torch.nn as nn


class FrobNorm(nn.Module):
    def __init__(self, margin: float=1.0):
        super(FrobNorm, self).__init__()
        self.D = margin
        
    def forward(self, input, target):
        x = (torch.norm((target - input), p=None))**2
        #x = x/2*self.D
        return x