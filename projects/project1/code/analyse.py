import matplotlib as plt
import numpy as np

def normalise(dataset, max_values, min_values):
    for i in range(dataset.shape[1]):
        dataset[:, i] = np.vectorize(lambda x: (x - min_values[i]) / (max_values[i] - min_values[i]))(dataset[:, i])
    return dataset

def norm_data(dataset):
    min_values = np.min(dataset, axis=0)
    max_values = np.max(dataset, axis=0)
    dataset = normalise(dataset, max_values, min_values)
    # for each column in the dataset, do the average
    n,m = dataset.shape
    norm = np.zeros(m)
    for i in range(n):
        norm[i]=np.mean(dataset[i,:])
    return norm

def compare_plot(datatest, y_test, y_pred):
    norm = norm_data(datatest)
    plt.plot(norm, y_test, 'ro')
    plt.plot(norm, y_pred, 'bo')
    plt.show()


