import numpy as np
import activation
from loss import cross_entropy
import helper
from sklearn.metrics import accuracy_score


class MLP:
    def __init__(self, input_size, output_size, hidden1, hidden2, hidden3, epochs, l_rate=0.01):
        self.lr = l_rate
        self.input_size = input_size
        self.epochs = epochs
        
        self.n = 0
        
        self.weights = {
            'w1': self.init_weights(input_size, hidden1),
            'w2': self.init_weights(hidden1, hidden2),
            'w3': self.init_weights(hidden2, hidden3),
            'w4': self.init_weights(hidden3, output_size)
        }
        
        self.biases = {
            'b1': np.zeros((1, hidden1)),
            'b2': np.zeros((1, hidden2)),
            'b3': np.zeros((1, hidden3)),
            'b4': np.zeros((1, output_size))
        }
        
    def init_weights(self, input_size, output_size):
        var = np.sqrt(2/(input_size))
        return np.random.normal(scale=var, size=(input_size, output_size))
    
    def f_propagation(self, X): 
        self.n += 1 
        self.lc1 = np.dot(X, self.weights['w1']) + self.biases['b1']
        self.a1 = activation.leaky_relu(self.lc1)
        
        self.lc2 = np.dot(self.a1, self.weights['w2']) + self.biases['b2']
        self.a2 = activation.leaky_relu(self.lc2)
        
        self.lc3 = np.dot(self.a2, self.weights['w3']) + self.biases['b3']
        self.a3 = activation.leaky_relu(self.lc3)
        
        self.lc4 = np.dot(self.a3, self.weights['w4']) + self.biases['b4']
        self.a4 = activation.softmax(self.lc4)
        return self.a4
        
    def b_propagation(self, X, y):
        m = X.shape[0]
        
        dlc4 = self.a4 - y
        dw4 = np.dot(self.a3.T, dlc4) / m
        db4 = np.sum(dlc4, axis=0, keepdims=True) / m

        da3 = np.dot(dlc4, self.weights['w4'].T)
        dlc3 = da3 * activation.leaky_relu_derivative(self.lc3)
        dw3 = np.dot(self.a2.T, dlc3) / m
        db3 = np.sum(dlc3, axis=0, keepdims=True) / m

        da2 = np.dot(dlc3, self.weights['w3'].T)
        dlc2 = da2 * activation.leaky_relu_derivative(self.lc2)
        dw2 = np.dot(self.a1.T, dlc2) / m
        db2 = np.sum(dlc2, axis=0, keepdims=True) / m

        da1 = np.dot(dlc2, self.weights['w2'].T)
        dlc1 = da1 * activation.leaky_relu_derivative(self.lc1)
        dw1 = np.dot(X.T, dlc1) / m
        db1 = np.sum(dlc1, axis=0, keepdims=True) / m
        
        dw1, dw2, dw3, dw4 = self.clip_gradients((dw1, dw2, dw3, dw4))
        
        self.weights['w4'] -= self.lr * dw4
        self.biases['b4'] -= self.lr * db4
        self.weights['w3'] -= self.lr * dw3
        self.biases['b3'] -= self.lr * db3
        self.weights['w2'] -= self.lr * dw2
        self.biases['b2'] -= self.lr * db2
        self.weights['w1'] -= self.lr * dw1
        self.biases['b1'] -= self.lr * db1
        
    
    def fit(self, X, y, X_val=None, y_val=None, batch_size=50):
        X = X.astype(np.float32)
        y_onehot, self.one_hot_mapping = helper.one_hot_encode(y)

        if X_val is not None and y_val is not None:
            X_val = X_val.astype(np.float32)
            y_val_onehot, y_val_map = helper.one_hot_encode(y_val)

        m = X.shape[0]
        self.train_losses = []
        self.val_losses = []

        for epoch in range(self.epochs):
            print(self.n)
            indices = np.random.permutation(m)
            X_shuffled, y_shuffled = X[indices], y_onehot[indices]
            
            batch_losses = []

            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                y_pred = self.f_propagation(X_batch)
                loss = cross_entropy(y_batch, y_pred)
                batch_losses.append(loss)

                self.b_propagation(X_batch, y_batch)

            mean_train_loss = np.mean(batch_losses)
            self.train_losses.append(mean_train_loss)

            if X_val is not None:
                y_val_pred = self.f_propagation(X_val)
                val_loss = cross_entropy(y_val_onehot, y_val_pred)
                self.val_losses.append(val_loss)
                print(f"iter {epoch}: loss = {loss:.4f}, val_loss = {val_loss:.4f}")
        
            
    def predict(self, X):
        y_pred = self.f_propagation(X)
        y_pred = helper.one_hot_decode(y_pred, self.one_hot_mapping)
        return y_pred
       
    def clip_gradients(self, grads, max_norm=1.0):
        total_norm = 0
        for g in grads:
            total_norm += np.sum(np.square(g))
        total_norm = np.sqrt(total_norm)

        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for g in grads:
                g *= scale
        return grads
        