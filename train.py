from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from model import DeepSemiNMF
from losses import FrobNorm
import sklearn
from scipy.io import loadmat
from sklearn.cluster import KMeans
import numpy as np
from helpers import load_data, evaluate_nmi

PATH = "data/PIE_pose27.mat"
data = load_data(PATH)

n_classes = np.unique(gnd).shape[0]
kmeans = KMeans(n_classes, precompute_distances=False)

#------------- Hyper Parameters --------------#
EPOCHS = 1000
LR = 5e-8
layers = [400, 100]

#------------- Define Model -----------------#
model = DeepSemiNMF(torch.tensor(data), layers)
criterion = FrobNorm()
optimizer = optim.Adam(model.params, lr=LR)

#---------------- Train --------------------#
display_in = 50

for epoch in range(EPOCHS):
  X_pred = model(torch.tensor(data))
  loss = criterion(X_pred, torch.tensor(data.T))
  loss.backward()
  optimizer.step()
  if epoch % display_in == 0:
      print(f'Epoch {epoch} ---------- Loss {loss}') 
      
#-------------- Evaluate -------------------#
with torch.no_grad():
  H = model.params[-1].detach()
  obj = nn.ReLU()
  H = obj(H)

#--------------------- Calculate Normalzed Mutual Information Score -----------------------------------------#  
print("K-means clustering using the Semi-NMF features has an NMI of {:.2f}%".format(100 * evaluate_nmi(H.T)))