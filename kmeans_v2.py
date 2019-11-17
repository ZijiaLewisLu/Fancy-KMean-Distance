import numpy as np
from .gd import gradient_descent, minibatch_gradient_descent #newton_raphson
from sklearn.cluster.k_means_ import _k_init
from .kmeans import distance as compute_distance
import time

def kmeans(method, X, C, p, 
        km_tol=1e-2, gd_tol=1e-3, initial_gd_step_size=0.05, num_reduction=3, 
        batch_size=512):
    """
    only support gradient_descent and minibatch_gradient_descent for now
    """

    squared_norm = (X**2).sum(1)
    centers = _k_init(X, C, squared_norm, np.random.RandomState())

    dist = np.zeros([len(X), C])
    for i, c in enumerate(centers):
        dist[:, i] = compute_distance(c, X, p)
    assign = dist.argmin(1)

    stop = False
    km_ct = 0
    prev_mse = 1000
    reduce_count = 0
    gd_step_size = initial_gd_step_size
    cumu_difference = []
    while not stop:
            
        # print("KM Iteration", km_ct)
        # b = time.time()
        # update centers
        total_mse = 0
        new = []
        num_non_empty = 0
        for I in range(C):
            mask = assign == I
            # print(I, "Number of Points", mask.sum())
            if mask.sum() == 0:
                continue # skip empty cluster
            num_non_empty += 1

            if mask.sum() == 1: # cluster of only one point
                # print("Only One Point, Skip")
                newc = X[mask]
                new.append([I, newc])
                continue

            x0 = X[mask]
            c = centers[I]
            if method == "gd":
                newc, mse, ct, diff = gradient_descent(c, x0, p, 
                        step_size=gd_step_size, max_step=2000, eps=gd_tol)
            elif method == "sgd":
                newc, mse, ct, diff = minibatch_gradient_descent(c, x0, p, batch_size=batch_size,
                        step_size=gd_step_size, max_step=2000, eps=gd_tol)

            new.append([I, newc])
            total_mse += mse

        for i, c in new:
            centers[i] = c

        # compute new distance and assignment
        dist = np.zeros([len(X), C])
        for i, c in enumerate(centers):
            dist[:, i] = compute_distance(c, X, p)
        assign = dist.argmin(1)

        # record difference
        total_mse = total_mse / num_non_empty
        abs_diff = prev_mse - total_mse
        cumu_difference.append(abs_diff)
        prev_mse = total_mse

        # stop criterion
        diff_mean = np.abs( np.mean(cumu_difference[-8:]) )
        if ( diff_mean < km_tol ): # and assign_diff < 0.001 :
            print( cumu_difference[-8:] )
            if reduce_count >= num_reduction:
                stop = True
                print("Reached Reduce 3 times, Breakout")
            print("Update Small Enough, Reduce GD Step Size") # reduce learning rate to improve
            gd_step_size = gd_step_size / 5
            gd_tol = gd_tol / 5
            km_tol = km_tol / 5
            reduce_count += 1

        km_ct += 1

        # e = time.time()
        # print("Duration", (e-b)/60)
   
    mse = 0
    for i in range(C):
        mask = assign == I
        mse += dist[mask, I].mean()
    mse = mse / C

    return centers, mse, assign, ct
