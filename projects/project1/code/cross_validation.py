import numpy as np
from dask.array.random import logistic

from implementations import *
from build_poly import *

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)
    """

    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train:
    x_train = np.copy(x)
    y_train = np.copy(y)

    x_test = np.zeros(k_indices.shape[1])
    y_test = np.zeros(x_test.shape)
    x_train = np.delete(x_train, k_indices[k])
    y_train = np.delete(y_train, k_indices[k])

    j = 0
    for i in k_indices[k]:
        x_test[j] = x[i]
        y_test[j] = y[i]
        j += 1

    x_train = build_poly(x_train, degree)
    x_test = build_poly(x_test, degree)

    w = ridge_regression(y_train, x_train, lambda_)
    loss_tr = np.sqrt(2 * compute_loss(y_train, x_train, w))
    loss_te = np.sqrt(2 * compute_loss(y_test, x_test, w))

    return loss_tr, loss_te

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def general_cross_validation(y, x, k_indices, k, function, args):
    """return the loss of function for a fold corresponding to k_indices

    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        function:     the tested function

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)
    """

    # ***************************************************
    # INSERT YOUR CODE HERE
    # get k'th subgroup in test, others in train:
    x_train = np.copy(x)
    y_train = np.copy(y)

    x_test = np.zeros((k_indices.shape[1], x_train.shape[1]))
    y_test = np.zeros(x_test.shape[0])
    x_train = np.delete(x_train, k_indices[k], 0)
    y_train = np.delete(y_train, k_indices[k], 0)

    j = 0
    for i in k_indices[k]:
        x_test[j] = x[i]
        y_test[j] = y[i]
        j += 1
    w = np.zeros(x.shape[1])
    args.insert(0, w)
    args.insert(0, x_train)
    args.insert(0, y_train)
    w = function(*args)[0]
    print("************************************w*************************")
    print(w)
    loss_tr = compute_loss(y_train, x_train, w)
    loss_te = compute_loss(y_test, x_test, w)

    return loss_tr, loss_te


def logistic_cross_validation(y, x, k_indices, k, max_iter, gamma):

    x_train = np.copy(x)
    y_train = np.copy(y)

    x_test = np.zeros((k_indices.shape[1], x_train.shape[1]))
    y_test = np.zeros(x_test.shape[0])
    x_train = np.delete(x_train, k_indices[k], 0)
    y_train = np.delete(y_train, k_indices[k], 0)

    j = 0
    for i in k_indices[k]:
        x_test[j] = x[i]
        y_test[j] = y[i]
        j += 1
    w = np.zeros(x.shape[1])

    w, loss = logistic_regression(y_train, x_train,w,max_iter,gamma)

    loss_te = compute_logistic_loss(y_test, x_test, w)

    return loss, loss_te
