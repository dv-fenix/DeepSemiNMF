from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
from torch.autograd import Variable
from layers import SemiNMFLayer

#----set seed-----#
np.random.seed(1)
torch.manual_seed(1)
torch.autograd.set_detect_anomaly(True)

class DeepSemiNMF(nn.Module):
    def __init__(self, data, layers: list, pretrain: bool=True):
        #Constructor
        super(DeepSemiNMF, self).__init__()
        
        assert len(layers) > 0, "Number of layers need to be greater than 0."
      
        H = data.T
        self.params = []
        
        for i, rank in enumerate(layers, start=1):
            print(f'Pretraining {i}th layer with rank {rank}')
            W, H = self.__init_deconstructs__(H, rank, use_svd=pretrain)
            self.params.append(Variable(W, requires_grad=True))
            
        self.params.append(Variable(H, requires_grad=True)) 
        
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.1)
        
    def forward(self, X):
    
        H = self.relu(self.params[-1])
        
        for w in reversed(self.params[1:-1][:]):
            H = self.relu(torch.matmul(w, H))

        H = torch.matmul(self.params[0], H)
            
        return H
    
    def __init_deconstructs__(self, X, rank, use_svd: bool=True):
        if use_svd:
            return SemiNMFLayer.apply(X, rank)
            
        x, y = X.shape
        W = 0.08 * torch.FloatTensor(x, rank).uniform_()
        H = 0.08 * torch.FloatTensor(rank, y).uniform_()
        
        return W, H