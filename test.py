from scipy.io import loadmat
from sklearn.cluster import KMeans
import sklearn
import torch
from layers import SemiNMFLayer
import numpy as np
from helpers import load_data, evaluate_nmi

PATH = "data/PIE_pose27.mat"
data = load_data(PATH)

n_classes = np.unique(gnd).shape[0]
kmeans = KMeans(n_classes, precompute_distances=False)

#-------------- SEMI NMF -----------------#
X = torch.tensor(data.T)
#-------------- X ~ WH -------------------#
W, H = SemiNMFLayer.apply(X, 100) #--- Input: [num_features, num_samples] ---#

#--------------------- Calculate Normalzed Mutual Information Score -----------------------------------------#
print("K-means clustering using the Semi-NMF features has an NMI of {:.2f}%".format(100 * evaluate_nmi(H.T, kmeans)))

