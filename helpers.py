from scipy.io import loadmat
from sklearn.cluster import KMeans
import sklearn
import torch
import numpy as np

def load_mat(path):
    mat = loadmat(PATH, struct_as_record=False, squeeze_me=True)

    data, gnd = mat['fea'].astype('float32'), mat['gnd']

    # Normalise each feature to have an l2-norm equal to one.
    data /= np.linalg.norm(data, 2, 1)[:, None]
    
    return data
    
    

def evaluate_nmi(X, kmeans):
    pred = kmeans.fit_predict(X)
    score = sklearn.metrics.normalized_mutual_info_score(gnd, pred)
    
    return score