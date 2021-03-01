from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Function
from scipy.sparse.linalg import svds

class SemiNMFLayer(Function):
    
    @staticmethod
    def forward(ctx, X, rank: int=100):
        
        U, S, V = torch.svd(X, compute_uv=True, some=False)
        
        S = torch.diag(S[0:(rank-1)])
        U = torch.matmul(U[:, 0:(rank-1)], S)
        V = torch.transpose(V, 0, 1)[0:(rank-1), :]
        
        x, y = X.shape
        
        Unew = U[:, 0]
        Vnew = V[0, :]
        
        __U = torch.where(torch.less(torch.min(V[0, :]), torch.min(-V[0, :])), -(Unew.view(x, 1)), Unew.view(x, 1))
        __V = torch.where(torch.less(torch.min(V[0, :]), torch.min(-V[0, :])), -(Vnew.view(1, y)), Vnew.view(1, y))
        if rank > 2:
            for i in range(1,rank-1):
                Unew = Unew.view(x, 1)
                Vnew = Vnew.view(1, y)
                __U = torch.where(torch.less(torch.min(V[0, :]), torch.min(-V[0, :])), torch.cat((__U, -Unew), dim=1), torch.cat((__U, Unew), dim=1))
                __V = torch.where(torch.less(torch.min(V[0, :]), torch.min(-V[0, :])), torch.cat((__V, -Vnew), dim=0), torch.cat((__V, Vnew), dim=0))

            
            
        if rank == 2:
            A = torch.cat((U, -U), dim=1)
        else:
            Un = torch.transpose(-(torch.sum(U, dim=1)), 0, -1).view(x, 1)
            A = torch.cat((U, Un), dim=1)

        B = torch.cat((V, torch.zeros((1, y))), dim=0)

        if rank >= 3:
            b, _ = torch.min(V, dim=0)
            B = torch.subtract(B, torch.minimum(torch.tensor(0.), b))
        else:
            B = torch.subtract(B, torch.minimum(torch.tensor(0.), V))
        x = torch.tensor(x)
        y = torch.tensor(y)
        normalize = torch.sqrt(torch.multiply(x,y).type(torch.FloatTensor))
        norm = torch.norm(A)

        return torch.multiply(torch.div(A, norm),normalize), torch.div(torch.multiply(B, norm),normalize)
        