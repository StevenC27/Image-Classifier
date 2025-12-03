import numpy as np

# returns the output of leaky_relu activation applied to x.
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# returns the output of leaky_relu_derivative applied to x.
def leaky_relu_derivative(x, alpha=0.01):
    grad = np.ones_like(x)
    grad[x < 0] = alpha
    return grad

# returns the output of softmax activation applied to x. 
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    prob = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    return prob
