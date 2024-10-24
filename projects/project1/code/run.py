import runpy as np

import numpy

from projects.project1.helpers import load_csv_data

x_train, x_test, y_train, train_ids, test_ids = load_csv_data("/Users/deschryver/PycharmProjects/ML_course/projects/project1/data/dataset")

#on remplace les nan par des 0
x_train = numpy.nan_to_num(x_train)
x_test = numpy.nan_to_num(x_test)
y_train = numpy.nan_to_num(y_train)

