import numpy as np

# returns the cross entropy loss of y_true and y_pred.
def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred + 1e-8)) / y_true.shape[0]

# returns the derivative of the cross entropy loss of y_true and y_pred.
def cross_entropy_derivative(y_true, y_pred):
    return y_pred - y_true