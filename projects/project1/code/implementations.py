import numpy as np
from compute_gradient import *
from compute_loss import *
from batch_iter import *

def least_squares(y, tx):

    N = tx.shape[0]
    gram_matrix = tx.T @ tx
    w = np.linalg.solve(gram_matrix, tx.T @ y)

    error = y - (tx @ w)
    loss = 1/(2*N)*error.dot(error)

    return w, loss

#Linear regression using gradient descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad

    loss = compute_loss(y, tx, w)
    return loss, w

def ridge_regression(y,tx,lambda_):

    N = tx.shape[0]
    D = tx.shape[1]
    gram_matrix = (tx.T @ tx + lambda_*2*N*np.identity(D))
    w = np.linalg.solve(gram_matrix, tx.T @ y)

    error = y - (tx @ w)
    loss = 1 / (2 * N) * error.dot(error)

    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for n_iter in range(max_iters):
        batch = next(batch_iter(y, tx, 1))
        w = w - gamma * compute_gradient(batch[0], batch[1], w)

    loss = compute_loss(y, tx, w)
    return loss, w
