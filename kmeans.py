import numpy as np
from gd import gradient_descent, newton_raphson
from sklearn.cluster.k_means_ import _k_init # kmeans++ initialization
import time

def distance(arr_x, mat_y, p):
    n, h = mat_y.shape
    x = np.tile(arr_x, (n, 1)) # become nxh
    x_big = x >= mat_y
    big_sum = (x_big * (x-mat_y)).sum(axis=1) # n
    small_sum = ((1-x_big) * (mat_y-x)).sum(axis=1)
    sum_ = (big_sum ** p + small_sum ** p)
    dist = sum_ ** (1/p)
    return dist

def kmeans(data, K, p, method="gd", eps=1e-4, step_size=0.1, rs=None):

    if method == "gd":
        find_center = gradient_descent
    elif method == "nr":
        find_center = newton_raphson

    squared_norm = (data ** 2).sum(axis=1)
    centers = _k_init(data, K, squared_norm, np.random.RandomState(rs))
    diff = 100
    prev = -10
    ct = 0

    # all_niters = []
    while diff > eps:
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
            new_c, se, niter, diff = find_center(init_c, d, p, 
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

    return centers, average_mse, assign

MP_X = None

def _mp_gd_helper(method, p, cid, init, cluster_mask, kwargs):
    """
    compute centroid and distance of all data point w.r.t the distance
    Input:
        method: gd or nr
        p: distance parameter
        cid: id of centroid
        init: initialization for centroid
        X: all data point
        cluster_mask: mask showing the data belong to the cluster
        kwargs: dict for gradient_descent function kwargs

    Return:
        cid, 
        updated centroid, 
        all data's distance to the updated centroid, 
        ( number of iterations, average of intra cluster distance ) 

        the last tuple is to help monitor gradient_descent performance
    """
    global MP_X 
    if method == "gd":
        find_center = gradient_descent
    elif method == "nr":
        find_center = newton_raphson

    if cluster_mask.sum() == 0:
        Xdist = distance(init, MP_X, p)
        return (cid, init, Xdist, (0, 0, 0))
    
    data = MP_X[cluster_mask]
    new_c, mdist, niter, diff = find_center(init, data, p, **kwargs)
    Xdist = distance(new_c, MP_X, p)

    return (cid, new_c, Xdist, (niter, mdist.sum()))

def kmeans_mp(m, data, K, p, method="gd", eps=1e-4, step_size=0.1, max_step=2000, rs=None):
    """
    multiprocess kmeans
    m: number of process to create
    """
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    global MP_X
    MP_X = data

    pool = mp.Pool(processes=m)

    squared_norm = (data ** 2).sum(axis=1)
    centers = _k_init(data, K, squared_norm, np.random.RandomState(rs))
    diff = 100
    prev = -10
    ct = 0

    # all_niters = []
    kwargs = {"eps":eps, "step_size":step_size, "max_step":max_step}

    # initalize distance for the first run
    dist = []
    for k in range(K):
        d = distance(centers[k], data, p)
        dist.append(d)
    assign = np.stack(dist, axis=0).argmin(axis=0)

    while diff > eps:
        ct += 1
        # print('iter', ct)
        average_mse = 0.0 

        # for each cluster, create input to _mp_gd_helper function 
        inp = []
        for k in range(K):
            mask = assign == k
            init = data[mask].mean(axis=0)
            d = [method, p, k, init, mask, kwargs]
            inp.append(d)
        
        # list of output from _mp_gd_helper
        rslt = pool.starmap(_mp_gd_helper, inp)

        dist = []
        niters = []
        mse = 0.0
        for cid, new_c, Xdist, info in rslt:
            centers[cid] = new_c
            dist.append(Xdist)
            niters.append(info[0])
            mse += info[1]

            if np.isnan(new_c).any():
                print("Nan!!", k, new_c)

        assign = np.stack(dist, axis=0).argmin(axis=0)

        average_mse = mse / data.shape[0] # average intra-cluster distance, similar to mean square error for euclidean distance 
        diff = np.abs(average_mse-prev) 
        prev = average_mse

    pool.close()

    return centers, average_mse, assign

if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    random_state = 170
    n_samples = 1000
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)

    centers, dist, assign = kmeans(X_aniso, 3, 2, method='nr')
    print(centers)
