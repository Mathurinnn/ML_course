import numpy as np

def least_squares(y, tx):

    N = tx.shape[0]
    gram_matrix = tx.T @ tx
    print(gram_matrix.shape)
    w = np.linalg.solve(gram_matrix, tx.T @ y)

    error = y - (tx @ w)
    loss = 1/(2*N)*error.dot(error)

    return w, loss