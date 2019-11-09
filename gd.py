import numpy as np

def distance_and_gradient_vector(arr_x, arr_y, p):
    """compute distance of vector x, y and the gradient of x,y w.r.t the distance"""
    x_big = arr_x >= arr_y
    big_sum = (x_big * (arr_x - arr_y)).sum()
    small_sum = ((1-x_big) * (arr_y-arr_x)).sum()
    sum_ = (big_sum ** p + small_sum ** p)
    dist = sum_ ** (1/p)
 
    gterm = sum_**(1/p -1)
    xterm = (big_sum ** (p-1))*x_big - (small_sum ** (p-1))*(1-x_big)
    xgrad = gterm * xterm
    yterm = -(big_sum ** (p-1))*x_big + (small_sum ** (p-1))*(1-x_big)
    ygrad = gterm * yterm

    return dist, xgrad, ygrad

def gradient_descent_vector(arr_x, mat_y, p, step_size=0.01):
    """
    use distance_and_gradient_vector to find the best x
    minimize the average (not sum) of distances between x and every row of Y
    """
    diff = 100
    n = mat_y.shape[0] 
    prev = -10

    while diff > 1e-3:
        dist = 0
        gx = 0
        for i in range(n):
            d, g, _ = distance_and_gradient_vector(arr_x, mat_y[i], p)
            dist += d
            gx += g
        gx = gx / n
        dist = dist / n

        arr_x = arr_x - step_size * gx
        diff = np.abs(dist-prev)
        prev = dist

    return arr_x, dist

def distance_and_gradient(arr_x, mat_y, p):
    """
    compute distance between vector x and each row of matrix y
    and the gradient of x to minimize the average distance
    """
    n, h = mat_y.shape
    x = np.tile(arr_x, (n, 1)) # become nxh
    x_big = x >= mat_y
    big_sum = (x_big * (x-mat_y)).sum(axis=1) # n
    small_sum = ((1-x_big) * (mat_y-x)).sum(axis=1)
    sum_ = (big_sum ** p + small_sum ** p)
    dist = sum_ ** (1/p)
 
    sum_ = np.maximum(sum_, 1e-5) # avoid divided by zero
    gterm = (sum_**(1/p -1)).reshape(n, 1)
    xterm = (big_sum ** (p-1)).reshape(n, 1)*x_big - (small_sum ** (p-1)).reshape(n, 1)*(1-x_big) # nxh
    xgrad = (gterm * xterm).mean(axis=0) # h

    return dist, xgrad

def gradient_descent(arr_x, mat_y, p, eps=1e-3, step_size=0.01, max_step=2000):
    """
    gradient descent method to find the optimal x
    """
    diff = 100
    prev = -10

    ct = 0
    while diff > eps:
        dist, gx = distance_and_gradient(arr_x, mat_y, p)
        arr_x = arr_x - step_size * gx
        diff = np.abs(dist.mean()-prev)
        prev = dist.mean()
        ct += 1
        if ct > max_step:
            break
    return arr_x, dist, ct, diff

def distance_grad_hessian(arr_x, mat_y, p):
    """
    compute distance, gradient, hessian
    grad, hessain computed based on average distances 
    """
    n, h = mat_y.shape
    x = np.tile(arr_x, (n, 1)) # n,h
    x_big = x >= mat_y
    B = (x_big * (x-mat_y)).sum(axis=1) # n
    S = ((1-x_big) * (mat_y-x)).sum(axis=1) # n
    sum_ = (B ** p + S ** p) # n
    dist = sum_ ** (1/p)
 
    sum_ = np.maximum(sum_, 1e-5) # avoid divided by zero
    gcommon = (sum_**(1/p -1)).reshape(n, 1)
    grad = (B ** (p-1)).reshape(n, 1)*x_big - (S ** (p-1)).reshape(n, 1)*(1-x_big) # nxh
    grad = (gcommon * grad).mean(axis=0) # h

    combine = lambda x, y : x.reshape(n, h, 1)*y.reshape(n, 1, h)
    pp = combine(x_big, x_big)     # n, h, h.  pp[i,j,k] if x[j] >= y[i,j] and x[k] >= y[i,k]
    pn = combine(x_big, 1-x_big)   # n, h, h.  pp[i,j,k] if x[j] >= y[i,j] and x[k] <  y[i,k]
    np_= combine(1-x_big, x_big)   # n, h, h.  pp[i,j,k] if x[j] <  y[i,j] and x[k] >= y[i,k]
    nn = combine(1-x_big, 1-x_big) # n, h, h.  pp[i,j,k] if x[j] <  y[i,j] and x[k] <  y[i,k]

    common = ((1-p) * ((sum_)**(1/p - 1))).reshape(n, 1, 1)
    pp_term = (B**(2*p-2)/sum_ - B**(p-2)).reshape(n, 1, 1)
    nn_term = (S**(2*p-2)/sum_ - S**(p-2)).reshape(n, 1, 1)
    pn_term = np_term = (-1 * S**(p-1) * B **(p-1) / sum_).reshape(n, 1, 1)

    H = pp*pp_term + pn*pn_term + np_*np_term + nn*nn_term
    H = (common*H).mean(axis=0) # n,h,h -> h,h

    return dist, grad, H

def newton_raphson(x, y, p, eps=1e-3, step_size=0.1, max_step=2000):
    """
    newton raphson method to find the optimal x
    """
    diff = 100
    prev = -10

    ct = 0
    while diff > eps:
        dist, g, H = distance_grad_hessian(x, y, p)
        H_inv = np.linalg.pinv(H)
        x = x - step_size*H_inv.dot(g)
        diff = np.abs(dist.mean()-prev)
        prev = dist.mean()
        ct += 1
        if ct > max_step:
            break

    return x, dist, ct, diff


if __name__ == '__main__':
    y = np.array(
        [[0, 1, 1],
        [2, 0, 0]])
    x = np.array([0, 0, 0])

    p = 2

    nx, d = gradient_descent_vector(x, y, p)
    print(nx)
    print(d)
    nx, d, ct = gradient_descent(x, y, p)
    print(nx)
    print(d.mean(), ct)
    nx, d, ct = newton_raphson(x, y, p)
    print(nx)
    print(d.mean(), ct)
