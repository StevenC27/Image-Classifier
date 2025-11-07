import numpy as np

def mle_loss(y_true, y_pred):
    n = y_true.shape[0]
    return (1/n) * sum((y_true-y_pred)**2)


def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]


def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true