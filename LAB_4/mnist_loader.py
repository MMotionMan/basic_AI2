from pickle import *
import gzip
from matplotlib import pyplot

import numpy as np


def load_data():
    f = gzip.open('/Users/anatoliy/PycharmProjects/basic_AI2/LAB_4/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = load(f, encoding='latin1')
    f.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return training_data, validation_data, test_data


def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

train_X, validation_X, test_X = load_data()
print(train_X[0])
for i in range(9):
    pyplot.subplot(360 + 1 + i)
    print(train_X[0][i])
    pyplot.imshow(train_X[0][i].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
    pyplot.show()
