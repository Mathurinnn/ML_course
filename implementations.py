import numpy as np

from projects.project1.code.batch_iter import batch_iter
from projects.project1.code.compute_gradient import compute_gradient, compute_gradient_logistic
from projects.project1.code.compute_loss import compute_loss


def least_squares(y, tx):

    n = tx.shape[0]
    gram_matrix = tx.T @ tx
    w = np.linalg.solve(gram_matrix, tx.T @ y)

    error = y - (tx @ w)
    loss = 1/(2*n)*error.dot(error)

    return w, loss

#Linear regression using gradient descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        w = w - gamma * grad

    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y,tx,lambda_):

    N = tx.shape[0]
    D = tx.shape[1]
    gram_matrix = (tx.T @ tx + lambda_*2*N*np.identity(D))
    w = np.linalg.solve(gram_matrix, tx.T @ y)

    error = y - (tx @ w)
    loss = 1 / (2 * N) * error.dot(error)

    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma): #
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w)
        w = w - gamma * grad
        # should there be a loss computation here?
    return w

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    w = initial_w

    for n_iter in range(max_iters):
        batch = next(batch_iter(y, tx, 1))
        w = w - gamma * compute_gradient(batch[0], batch[1], w)

    loss = compute_loss(y, tx, w)
    return loss, w
