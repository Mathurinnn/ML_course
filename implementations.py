import numpy as np

from projects.project1.code.batch_iter import batch_iter
from projects.project1.code.compute_gradient import compute_gradient, compute_gradient_logistic
from projects.project1.code.compute_loss import compute_loss, compute_logistic_loss


def least_squares(y, tx):
    """
       Computes the optimal weights using the least squares method.

       Inputs:
       y : numpy array
           The target values (labels).
       tx : numpy array
           The input features, where each row represents a sample and each column represents a feature.

       Outputs:
       w : numpy array
           The optimal weights that minimize the least squares loss.
       loss : float
           The computed loss associated with the optimal weights.
       """
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
    """
        regularized logistic regression using gradient descent.

        Inputs:
        y : numpy array
            The target values (labels).
        tx : numpy array
            The input features, where each row represents a sample and each column represents a feature.
        lambda_ : float
            The regularization parameter that controls the amount of regularization.
        initial_w : numpy array
            The initial weights for the logistic regression model.
        max_iters : int
            The maximum number of iterations for gradient descent.
        gamma : float
            The step size (learning rate) for gradient descent.

        Outputs:
        w : numpy array
            The weights that minimize the regularized logistic loss.
        loss : float
            The computed loss associated with the weights.
        """
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad

    loss = compute_logistic_loss(y, tx, w)
    return w, loss