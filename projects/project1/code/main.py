from projects.project1.helpers import load_csv_data
from projects.project1.code.implementations import *

if __name__ == '__main__':

    x_train, x_test, y_train, train_ids, test_ids = load_csv_data("C:\\Users\mathu\PycharmProjects\ML_course\projects\project1\data\dataset\dataset")

    print(least_squares(y_train, x_train))