import numpy as np

from projects.project1.helpers import load_csv_data
from projects.project1.code.implementations import *


MAX_ITERS = 2
GAMMA = 0.1

def test_mean_squared_error_gd(y, tx, initial_w):
    w, loss = mean_squared_error_gd(
        y, tx, initial_w, 2, 0.1
    )

    expected_w = np.array([-0.050586, 0.203718])
    expected_loss = 0.051534

    print(loss, expected_loss)
    print(w, expected_w)


def test_least_squares(y, tx):
    w, loss = least_squares(y, tx)

    expected_w = np.array([0.218786, -0.053837])
    expected_loss = 0.026942

    print(w, expected_w)
    print(loss, expected_loss)

def test_ridge_regression_lambda0(y, tx):
    lambda_ = 0.0
    w, loss = ridge_regression(y, tx, lambda_)

    expected_loss = 0.026942
    expected_w = np.array([0.218786, -0.053837])

    print(w, expected_w)
    print(loss, expected_loss)


def test_ridge_regression_lambda1(y, tx):
    lambda_ = 1.0
    w, loss = ridge_regression(y, tx, lambda_)

    expected_loss = 0.03175
    expected_w = np.array([0.054303, 0.042713])

    print(w, expected_w)
    print(loss, expected_loss)

def test_mean_squared_error_sgd( y, tx, initial_w):
    # n=1 to avoid stochasticity
    w, loss = mean_squared_error_sgd(
        y[:1], tx[:1], initial_w, MAX_ITERS, GAMMA
    )

    expected_loss = 0.844595
    expected_w = np.array([0.063058, 0.39208])

    print(w, expected_w)
    print(loss, expected_loss)

if __name__ == '__main__':

    #x_train, x_test, y_train, train_ids, test_ids = load_csv_data("C:\\Users\mathu\PycharmProjects\ML_course\projects\project1\data\dataset")

    print(test_mean_squared_error_sgd(np.array([0.1, 0.3, 0.5]), np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]]), np.array([0.5, 1.0])))