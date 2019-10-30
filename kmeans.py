import numpy as np
from gd import gradient_descent, newton_raphson
from sklearn.cluster.k_means_ import _k_init # kmeans++ initialization

def distance(arr_x, mat_y, p):
    n, h = mat_y.shape
    x = np.tile(arr_x, (n, 1)) # become nxh
    x_big = x >= mat_y
    big_sum = (x_big * (x-mat_y)).sum(axis=1) # n
    small_sum = ((1-x_big) * (mat_y-x)).sum(axis=1)
    sum_ = (big_sum ** p + small_sum ** p)
    dist = sum_ ** (1/p)
    return dist

def kmeans(data, K, p, method="gd", eps=1e-4, step_size=0.1):

    if method == "gd":
        find_center = gradient_descent
    elif method == "nr":
        find_center = newton_raphson

    squared_norm = (data ** 2).sum(axis=1)
    centers = _k_init(data, K, squared_norm, np.random.RandomState())
    diff = 100
    prev = -10
    ct = 0

    while diff > eps:
        ct += 1
        print('iter', ct)
        # compute assignment
        all_dist = []
        for k in range(K):
            dist = distance(centers[k], data, p)
            all_dist.append(dist)

        assign = np.stack(all_dist, axis=0).argmin(axis=0)

        # update centers
        average_dist = 0
        for k in range(K):
            d = data[assign==k] # data assigned to cluster k
            new_c, dist, niter = find_center(centers[k], d, p, 
                                    eps=eps, step_size=step_size) 
            centers[k] = new_c
            if np.isnan(new_c).any():
                print("Nan!!", k, new_c)
            average_dist += dist.sum()
        
        average_dist = average_dist / data.shape[0]
        diff = np.abs(average_dist-prev) 
        prev = average_dist

    return centers, average_dist, assign



if __name__ == "__main__":
    from sklearn.datasets import make_blobs
    random_state = 170
    n_samples = 1000
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)

    centers, dist, assign = kmeans(X_aniso, 3, 2, method='nr')
    print(centers)