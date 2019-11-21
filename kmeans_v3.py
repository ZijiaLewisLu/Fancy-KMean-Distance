import numpy as np
import os
from sklearn.cluster.k_means_ import _k_init # kmeans++ initialization
import time
from metrics import *
import torch
from torch.autograd import Variable
import torch.optim as optim
from collections import OrderedDict

def distance(arr_x, mat_y, p):
    n, h = mat_y.shape
    x = np.tile(arr_x, (n, 1)) # become nxh
    x_big = x >= mat_y
    big_sum = (x_big * (x-mat_y)).sum(axis=1) # n
    small_sum = ((1-x_big) * (mat_y-x)).sum(axis=1)
    sum_ = (big_sum ** p + small_sum ** p)
    dist = sum_ ** (1/p)
    return dist


class Find_Center(torch.nn.Module):
    def __init__(self, dim, p, norm=False):
        super(Find_Center, self).__init__()
        self.center = torch.nn.Parameter(torch.randn([dim]))
        self.dim = dim
        self.p = p
        self.norm = norm #Set norm to be True for normalized distance metric

    def distance(self, x):
        diff = x - self.center
        dist = (torch.sum(diff * (diff >= 0), axis=-1) ** self.p + torch.sum(-diff * (diff < 0), axis=-1) ** self.p) ** (1/self.p)

        if self.norm:
            c = self.center.expand_as(x)
            stack = torch.stack((x.abs(), c.abs(), diff.abs()), 0)
            m, _ = stack.max(0)
            s = m.sum(-1)
            dist /= s

        return dist

    def forward(self, x):
        dist = self.distance(x)

        return dist


def auto_find_center(arr_x, mat_y, p, model, eps=1e-6, step_size=0.01, max_step=2000):
    '''
      Returns:
          arr_x: center_point
          dist: distances
          ct: count of iterations
          diff: difference between loss of the final two iterations
    '''
    y = torch.from_numpy(mat_y).float().cuda()
    label = torch.from_numpy(np.zeros([y.shape[0]])).float().cuda()    

    # initialize center points
    new_state_dict = OrderedDict({'center': torch.from_numpy(arr_x)})
    model.load_state_dict(new_state_dict, strict=False)

    diff = 10000
    prev = 1000
    ct = 0

    best_loss = [1e6, 0]
    while diff / prev > eps and ct < max_step:
        dist = model(y)
        loss = loss_fn(dist, label)

        ### Early stop if mse stops decreasing for 10 iterations
        if loss.data.item() < best_loss[0]:
            best_loss = [loss.data.item(), ct]
        if ct > best_loss[1] + 10:
            print("finding center early stop!")
            break

        diff = abs(prev - loss.data.item())
        prev = loss.data.item()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ct += 1

    #print('{}: dist={}, diff={}'.format(ct, loss.data.item(), diff))

    arr_x = model.state_dict()['center'].data.cpu().numpy()
    dist = dist.cpu().detach().numpy()
    
    return arr_x, dist, ct, diff
       


def kmeans(data, K, p, model, eps=1e-4, step_size=0.1, rs=None):
    squared_norm = (data ** 2).sum(axis=1)
    centers = _k_init(data, K, squared_norm, np.random.RandomState(rs))
    diff = 10000
    prev = 1000
    ct = 0

    # all_niters = []
    best_mse = [1e6, 0]
    while diff / prev > eps and ct < 50:
        ct += 1
        # print('iter', ct)

        # compute assignment
        all_dist = []
        for k in range(K):
            dist = distance(centers[k], data, p)
            all_dist.append(dist)

        assign = np.stack(all_dist, axis=0).argmin(axis=0)

        # update centers
        average_mse = 0 # intra-cluster distance, similar to mean square error for euclidean distance
        track_niter = []
        for k in range(K):
            mask = assign==k
            if mask.sum() == 0: # skip empty cluster
                continue

            d = data[mask] # data assigned to cluster k

            init_c = d.mean(axis=0) # use mean to initialize
            new_c, se, niter, diff = auto_find_center(init_c, d, p, model,
                                    eps=eps, step_size=step_size) 
            
            centers[k] = new_c

            track_niter.append(niter)
            # all_niters.append(niter)
            if np.isnan(new_c).any():
                print("Nan!!", k, new_c)
            average_mse += se.sum()

        average_mse = average_mse / data.shape[0]
        diff = np.abs(average_mse-prev) 
        prev = average_mse
        #print('kmeans ct={}, diff={}, mse={}'.format(ct, diff, average_mse))

        ### Early stop if mse stops decreasing for 10 iterations
        if average_mse < best_mse[0]:
            best_mse = [average_mse, ct]
        if ct > best_mse[1] + 10:
            print("k-means early stop!")
            break

    return centers, average_mse, assign

if __name__ == "__main__":
    import h5py
    import scipy.sparse
    import scipy.io

    filepath = '../rehighdimensiondataset/housing.mat'
    try:
        f = h5py.File(filepath)
        X = scipy.sparse.csc_matrix((f['X']['data'], f['X']['ir'], f['X']['jc'])).toarray()
    except:
        f = scipy.io.loadmat(filepath)
        X = f['X']
    
    y = np.asarray(f['y']).reshape([-1])
    K = 2
    p = 2

    ### Initialize Find_Center Model, set norm=False for unnormalized distance metric, norm=True for normalized distance metric
    model = Find_Center(dim=X.shape[-1], p=p, norm=True).cuda()
    loss_fn = torch.nn.L1Loss(reduction='mean')
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for i in range(20):
        centers, dist, assign = kmeans(X, K, p, model)
        nmi, ari, acc, f1 = cluster_evaluate(y, assign, True)
