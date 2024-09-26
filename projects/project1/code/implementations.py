import numpy as np

def least_squares(y, tx):

    N = tx.shape[0]
    gram_matrix = tx.T @ tx
    w = np.linalg.solve(gram_matrix, tx.T @ y)

    error = y - (tx @ w)
    loss = 1/(2*N)*error.dot(error)

    return w, loss

#Linear regression using gradient descent
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
   
    n = np.shape(y)[0]#size of the dataset

    def mean_squared_error_gd_recursive(y, tx, w, max_iters, gamma):
        if(max_iters == 0):
            return (initial_w, (1/(2*n)) * np.sum(np.square(y - tx @ w)))
        else:
            grad = -(1/n) * (np.transpose(tx) @ (y - tx @ w))#gradient of the loss function
            return mean_squared_error_gd(y, tx, w - gamma * grad, max_iters - 1, gamma)
            
    return mean_squared_error_gd_recursive(y, tx, initial_w, max_iters, gamma)

def ridge_regression(y,tx,lambda_):

    N = tx.shape[0]
    D = tx.shape[1]
    gram_matrix = (tx.T @ tx + lambda_*2*N*np.identity(D))
    w = np.linalg.solve(gram_matrix, tx.T @ y)

    error = y - (tx @ w)
    loss = 1 / (2 * N) * error.dot(error)

    return w, loss