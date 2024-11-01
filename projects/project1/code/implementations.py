import numpy as np
from compute_gradient import *
from compute_loss import *
from batch_iter import *

def least_squares(y, tx):

    n = tx.shape[0]
    gram_matrix = tx.T @ tx
    w = np.linalg.solve(gram_matrix, tx.T @ y)

    error = y - (tx @ w)
    loss = 1/(2*n)*error.dot(error)

    return w, loss


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Mean squared error using gradient descent
    Returns the optimal weights and the loss
    y: labels
    tx: features
    initial_w: initial weights
    max_iters: number of iterations
    gamma: step size
    """
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad

    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y,tx,lambda_):
    """
    Ridge regression using normal equations
    Returns the optimal weights and the loss
    y: labels
    tx: features
    lambda_: regularization parameter
    """

    N = tx.shape[0]
    D = tx.shape[1]
    gram_matrix = (tx.T @ tx + lambda_*2*N*np.identity(D))
    w = np.linalg.solve(gram_matrix, tx.T @ y)

    error = y - (tx @ w)
    loss = 1 / (2 * N) * error.dot(error)

    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent
    Returns the optimal weights and the loss
    y: labels
    tx: features
    initial_w: initial weights
    max_iters: number of iterations
    gamma: step size
    """
    w = initial_w
    N = tx.shape[0]
    for n_iter in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w)
        w = w - gamma * grad
        # should there be a loss computation here?
    print(w)
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Mean squared error using stochastic gradient descent
    Returns the optimal weights and the loss
    y: labels
    tx: features
    initial_w: initial weights
    max_iters: number of iterations
    gamma: step size
    """
    w = initial_w

    for n_iter in range(max_iters):
        batch = next(batch_iter(y, tx, 1))
        w = w - gamma * compute_gradient(batch[0], batch[1], w)

    loss = compute_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    w = initial_w

    for n_iter in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad

    loss = compute_logistic_loss(y, tx, w)
    return w, loss

