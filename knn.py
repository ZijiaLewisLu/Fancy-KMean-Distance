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
from tqdm import tqdm

def load_sparse_matrix(db, shape):
    d = db["data"][()]
    ind = db["ir"][()]
    indp = db['jc'][()]
    
    m = csc_matrix((d, ind, indp), shape=shape)
    return m

def load_sparse_h5py(fname, shape):
    fp = h5py.File(fname, mode="r")
    X = load_sparse_matrix(fp["X"], shape)
    y = fp["y"][()]
    fp.close()
    return X, y

class Cosine():

    def pairwise(self, x, y):
        return cosine_distances(x, y)

class NewDistance():

    def __init__(self, p, normalize):
        self.p = p
        self.normalize = normalize

    def pairwise(self, arr_x, mat_y):
        """
        mat_x is of 1, D
        mat_y is of B, D
        """
        n, h = mat_y.shape
        x = np.tile(arr_x, (n, 1)) # become nxh
        x_big = x >= mat_y
        difference = x-mat_y
        big_sum = (x_big * difference).sum(axis=1) # n
        small_sum = ((1-x_big) * (-difference)).sum(axis=1)
        sum_ = (big_sum ** self.p + small_sum ** self.p)
        dist = sum_ ** (1/self.p)

        if self.normalize:
            stack = np.stack((x, mat_y, difference), 0)
            stack = np.abs(stack)
            m = stack.max(0).sum(-1)
            dist /= m
        return dist

def kfold_knn(X, y, metric, num_fold=5):
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
            for t in range(num_train-1):
                x1 = trainX[t:t+1, :] # 1, D
                x2 = trainX[t+1:,  :] # T-t, D
                # ipdb.set_trace()
                # print(x1.shape, x2.shape)
                dist = metric.pairwise(x1, x2)
                    
                distance_matrix[t, t+1:] = dist
                distance_matrix[t+1:, t] = dist
                
                kneighbor = distance_matrix[t].argsort()[:k].tolist()
                knn_label = trainy[kneighbor].tolist()
                ctr = Counter(knn_label)
                pred = ctr.most_common(1)[0][0]

                acc.append(pred == trainy[t])

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
            kneighbor = dist.argsort()[:best_k].tolist()
            try:
                knn_label = trainy[kneighbor].tolist()
            except:
                ipdb.set_trace()
            ctr = Counter(knn_label)
            pred = ctr.most_common(1)[0][0]
            acc.append( pred == testy[i] )

        fold_accuracy.append( np.mean(acc) )
        fold_best_k.append(best_k)

    return np.mean(fold_accuracy), fold_accuracy, fold_best_k

