import numpy as np
import h5py
from scipy.sparse import csc_matrix, issparse
from scipy.io import loadmat
from sklearn.neighbors import DistanceMetric
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_distances
from copy import copy
import ipdb
import os
from collections import Counter, OrderedDict
import torch
from tqdm import tqdm
from glob import glob

def load_sparse_matrix(db, shape):
    d = db["data"][()]
    ind = db["ir"][()]
    indp = db['jc'][()]
    
    m = csc_matrix((d, ind, indp), shape=shape)
    return m

def load_sparse_h5py(fname, shape):
    fp = h5py.File(fname, mode="r")
    X = load_sparse_matrix(fp["X"], shape)
    X = X.todense()
    y = fp["y"][()]
    fp.close()
    return X, y

class Cosine():

    def __init__(self):
        self._metric = torch.nn.CosineSimilarity(dim=1)

    def pairwise(self, x, y):
        return 1 - self._metric(x, y)

class Euclidean():

    def __init__(self):
        self._metric = torch.nn.PairwiseDistance(p=2.0, eps=1e-06)

    def pairwise(self, x, y):
        return self._metric(x, y)

class Manhattan():

    def __init__(self):
        self._metric = torch.nn.PairwiseDistance(p=1.0, eps=1e-06)

    def pairwise(self, x, y):
        return self._metric(x, y)

class Chebyshev():

    def pairwise(self, arr_x, mat_y):
        """
        mat_x is of 1, D
        mat_y is of B, D
        """
        x = arr_x.expand_as(mat_y)
        diff = torch.abs(mat_y - x)
        dist, _ = torch.max(diff, dim=1)
        return dist

class NewDistance():

    def __init__(self, p, normalize):
        self.p = p
        self.normalize = normalize

    def pairwise(self, arr_x, mat_y):
        """
        mat_x is of 1, D
        mat_y is of B, D
        """
        p = self.p
        x = arr_x.expand_as(mat_y)
        diff = mat_y - x
        bigger = (diff>=0).float()
        dist = (torch.sum(diff * bigger, axis=-1) ** p \
                + torch.sum(-diff * (1-bigger), axis=-1) ** p) ** (1/p)

        if self.normalize:
            stack = torch.stack((x.abs(), mat_y.abs(), diff.abs()), 0)
            m, _ = stack.max(0)
            s = m.sum(-1)
            dist /= s

        return dist

def kfold_knn(X, y, metric, num_fold=5, gpu=True):
    kf = KFold(n_splits=num_fold, shuffle=True)
    L = X.shape[0]
    
    fold_accuracy = []
    fold_best_k = []
    for train_idx, test_idx in kf.split(X):
        trainX, trainy = X[train_idx], y[train_idx]
        testX,  testy  = X[test_idx],  y[test_idx]

        # on train set, find the best K
        train_idx = list(train_idx)
        performance = []
        for k in tqdm(range(1, int(np.ceil(np.sqrt(L))), 2)):
            num_train = len(train_idx) 

            # leave one out test
            acc = []
            distance_matrix = np.zeros([num_train, num_train]) + np.Inf
            distance_matrix = torch.FloatTensor(distance_matrix)
            if gpu: 
                distance_matrix = distance_matrix.cuda()

            for t in range(num_train-1):
                x1 = trainX[t:t+1, :] # 1, D
                x2 = trainX[t+1:,  :] # T-t, D
                # ipdb.set_trace()
                # print(x1.shape, x2.shape)
                dist = metric.pairwise(x1, x2)
                    
                distance_matrix[t, t+1:] = dist
                distance_matrix[t+1:, t] = dist
                
                # ipdb.set_trace()
                
                # kneighbor_np = distance_matrix[t].argsort()[:k].tolist()
                kneighbor = torch.topk(distance_matrix[t], k, largest=False).indices
                knn_label = trainy[kneighbor].detach().cpu().numpy().tolist()
                ctr = Counter(knn_label)
                pred = ctr.most_common(1)[0][0]

                acc.append((pred == trainy[t]).item())

            acc = np.mean(acc)
            performance.append([k, acc])

        performance.sort(key=lambda x: x[1])
        best_k = performance[-1][0]
        # print('best_k', best_k)
        
        # classify using the best K
        # distance_matrix = np.zeros([len(test_idx), num_train])
        acc = []
        for i in range(len(test_idx)):
            tx = testX[i:i+1]
            dist = metric.pairwise(tx, trainX)
            # kneighbor = dist.argsort()[:best_k].tolist()
            kneighbor = torch.topk(dist, best_k, largest=False).indices
            knn_label = trainy[kneighbor].detach().cpu().numpy().tolist()
            ctr = Counter(knn_label)
            pred = ctr.most_common(1)[0][0]
            acc.append( (pred == testy[i]).item() )

        fold_accuracy.append( np.mean(acc) )
        fold_best_k.append(best_k)

    return np.mean(fold_accuracy), fold_accuracy, fold_best_k

def load_data():
    # dataset = ['pageblock.mat',
               # 'TTC3600.mat',
               # 'sdm06_twoclass.mat',
               # 'webkb.mat',
               # 'reuters.mat',
               # 'sdm06.mat',]

    fdr = '../data/high_dim_data/'
    data = {}

    h5py_sparse = [
        ['reuters.mat',[2065, 8943]],
        ['sdm06.mat', [930, 99899]],
        ['webkb.mat', [2803, 7288]],
        ['TTC3600.mat', [ 3600, 5692] ],
              ]
    # load h5py sparse matrix
    for d, shape in h5py_sparse:
        fname = fdr + d
        X, y = load_sparse_h5py(fname, shape)
        y = np.squeeze(y)
        data[d] = [X, y]

    # loadmat
    d = loadmat('../data/high_dim_data/sdm06_twoclass.mat')
    data["sdm06_twoclass.mat"] = [d["X"].todense(), np.squeeze(d["y"])]
    d = loadmat('../data/high_dim_data/pageblock2.mat')
    data["pageblock.mat"] = [d["X"], np.squeeze(d["y"])]

    return data


                



