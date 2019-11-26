import numpy as np
import os
from sklearn.cluster.k_means_ import _k_init # kmeans++ initialization
import time
from .metrics import cluster_evaluate
import torch
from torch.autograd import Variable
import torch.optim as optim
from collections import OrderedDict

def torch_dp(x, Y, p, norm=False):
    diff = Y - x
    bigger = (diff>=0).float()
    dist = (torch.sum(diff * bigger, axis=-1) ** p \
            + torch.sum(-diff * (1-bigger), axis=-1) ** p) ** (1/p)

    if norm:
        c = x.expand_as(Y)
        stack = torch.stack((c.abs(), Y.abs(), diff.abs()), 0)
        m, _ = stack.max(0)
        s = m.sum(-1)

        dist /= s

    return dist

class Find_Center(torch.nn.Module):
    def __init__(self, dim, p, norm=False):
        super(Find_Center, self).__init__()
        self.center = torch.nn.Parameter(torch.randn([dim]))
        self.dim = dim
        self.p = p
        self.norm = norm #Set norm to be True for normalized distance metric

    def forward(self, Y):
        dist = torch_dp(self.center, Y, self.p, self.norm)
        return dist


def gradient_descent_iteration(arr_x, mat_y, p, model, 
        eps=1e-6, optim_method="rprop", step_size=0.01, max_step=2000):
    '''
      Returns:
          arr_x: center_point
          dist: distances
          ct: count of iterations
          diff: difference between loss of the final two iterations
    '''
    # y = torch.from_numpy(mat_y).float().cuda()
    # label = torch.from_numpy(np.zeros([y.shape[0]])).float().cuda()    

    # initialize center points
    new_state_dict = OrderedDict({'center': arr_x})
    model.load_state_dict(new_state_dict, strict=False)

    diff = 10000
    prev = 1000
    ct = 0

    if optim_method == "adam":
        optimizer = optim.Adam(model.parameters(), lr=step_size)
    elif optim_method == "rprop":
        optimizer = optim.Rprop(model.parameters())

    best_loss = [1e6, 0]
    while diff / prev > eps and ct < max_step:
        dist = model(mat_y)
        loss = dist.mean()
        # loss = loss_fn(dist, label)

        ### Early stop if mse stops decreasing for 10 iterations
        if loss.data.item() < best_loss[0]:
            best_loss = [loss.data.item(), ct]
        # if ct > best_loss[1] + 10:
            # print(optim_method, "finding center early stop!")
            # break

        diff = abs(prev - loss.data.item())
        prev = max( loss.data.item(), 1e-5 )
        if prev == 0:
            import ipdb; ipdb.set_trace()
        model.zero_grad()
        loss.backward()
        optimizer.step()
        ct += 1

    arr_x = model.center.data
    
    return arr_x, dist, ct, diff
       


def kmeans(data, K, p, normed, eps=1e-4, 
        optim_method="rprop", step_size=0.001, 
        batch_size=None, mean_init=True,
        max_km_iteraton=50,
        rs=None, gpu=False):
    """
    if batch_size is not None, do Minibatch KMeans. (Not recommended, does not provide speedup)
    """

    model = Find_Center(dim=data.shape[-1], p=p, norm=normed)
    squared_norm = (data ** 2).sum(axis=1)
    centers = _k_init(data, K, squared_norm, np.random.RandomState(rs))
    diff = 10000
    prev = 1000
    ct = 0

    centers = torch.FloatTensor(centers)
    data = torch.FloatTensor(data)
    if gpu:
        centers = centers.cuda()
        data = data.cuda()
        model.cuda()

    best_mse = [1e6, 0]
    optim_difference = []
    while diff / prev > eps and ct < max_km_iteraton:
        ct += 1
        # print('iter', ct)

        if batch_size is not None:
            idx = np.random.choice(len(data), batch_size, replace=False)
            data_batch = data[idx]
        else:
            data_batch = data

        # compute assignment
        all_dist = []
        for k in range(K):
            dist = torch_dp(centers[k], data_batch, p, model.norm)
            all_dist.append(dist)

        value, assign = torch.stack(all_dist, 0).min(axis=0)

        # update centers
        average_mse = 0 # intra-cluster distance, similar to mean square error for euclidean distance
        track_niter = []
        for k in range(K):
            mask = assign==k
            if mask.sum() == 0: # skip empty cluster
                continue

            if mask.sum() == 1:
                centers[k] = data_batch[mask][0]
                continue

            d = data_batch[mask] # data assigned to cluster k

            if mean_init:
                init_c = d.mean(axis=0) # use mean to initialize
            else:
                init_c = centers[k]

            new_c, se, niter, diff = gradient_descent_iteration(init_c, d, p, model,
                                    optim_method=optim_method,
                                    eps=eps, step_size=step_size, max_step=3000) 
            centers[k] = new_c

            track_niter.append(niter)
            # all_niters.append(niter)
            if torch.isnan(new_c).any():
                print("Nan!!", k, new_c)
            average_mse += se.sum()

        average_mse = average_mse / data_batch.shape[0]
        diff = torch.abs(average_mse-prev) 
        prev = average_mse

        ### Early stop if mse stops decreasing for 10 iterations
        if average_mse < best_mse[0]:
            best_mse = [average_mse, ct]
        if ct > best_mse[1] + 10:
            print("k-means early stop!")
            break

    # if minibatch, compute all data distance
    if batch_size is not None:
        all_dist = []
        for k in range(K):
            dist = torch_dp(centers[k], data, p, model.norm)
            all_dist.append(dist)
        value, assign = torch.stack(all_dist, 0).min(axis=0)
        average_mse = value.mean()

    centers = centers.cpu().detach().numpy()
    average_mse = average_mse.item()
    assign = assign.cpu().detach().numpy()

    return centers, average_mse, assign

if __name__ == "__main__":
    import h5py
    import scipy.sparse
    import scipy.io
    from glob import glob
    import pickle as pk

    files = glob("small_dim_data/*.mat")
    for fname in files:
        f = scipy.io.loadmat(fname)
        X = f['X']
    
        y = np.asarray(f['y']).reshape([-1])
        K = 2
        p = 1
        normed = True

        print(fname)

        ### Initialize Find_Center Model, set norm=False for unnormalized distance metric, norm=True for normalized distance metric

        all_rslt = []
        compare = []
        simi_list = []
        for i in range(20):
            centers, mse_std, assign_std = kmeans(X, K, p, normed=normed, 
                    optim_method="rprop", batch_size=None, gpu=True)


        all_rslt.sort(key=lambda x: x[1])
        assign = all_rslt[0][2]
