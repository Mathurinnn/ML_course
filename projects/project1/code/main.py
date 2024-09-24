import numpy as np

from projects.project1.helpers import load_csv_data
from projects.project1.code.implementations import *

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


if __name__ == '__main__':

    #x_train, x_test, y_train, train_ids, test_ids = load_csv_data("C:\\Users\mathu\PycharmProjects\ML_course\projects\project1\data\dataset\dataset")


    print(test_least_squares(np.array([0.1, 0.3, 0.5]), np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])))
