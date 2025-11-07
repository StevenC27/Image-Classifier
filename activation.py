import numpy as np

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    return (x > 0).astype(float)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    grad = np.ones_like(x)
    grad[x < 0] = alpha
    return grad

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s(1 - s)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    prob = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return prob
