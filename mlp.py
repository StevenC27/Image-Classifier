import numpy as np
import activation
from loss import cross_entropy
import helper
from sklearn.metrics import accuracy_score

class MLP:
    def __init__(self, epochs, layers=[3072, 1024, 512, 256, 128, 5], lr=0.005, grad_clipping=True):
        self.lr = lr
        self.epochs = epochs
        self.n_layers = len(layers)-1
        self.grad_clipping = grad_clipping
        
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers):
            w = self.init_weights(layers[i], layers[i+1])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
    def init_weights(self, input_size, output_size):
        var = np.sqrt(2/(input_size))
        return np.random.normal(scale=var, size=(input_size, output_size))
    
    def f_propagation(self, X):
        self.l_combinations = []
        self.activations = [X]
        for i in range(self.n_layers-1):
            lc = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            a = activation.leaky_relu(lc)
            self.l_combinations.append(lc)
            self.activations.append(a)
            
        lc = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        a = activation.softmax(lc)
        self.l_combinations.append(lc)
        self.activations.append(a)
        return a
        
    def b_propagation(self, X, y):
        m = X.shape[0]
        dlc = self.activations[-1] - y
        dws = [None] * self.n_layers
        dbs = [None] * self.n_layers
        
        for i in reversed(range(self.n_layers)):
            a_prev = self.activations[i]
            dws[i] = np.dot(a_prev.T, dlc) / m
            dbs[i] = np.sum(dlc, axis=0, keepdims=True) / m
            if i != 0:
                da_prev = np.dot(dlc, self.weights[i].T)
                dlc_prev = da_prev * activation.leaky_relu_derivative(self.l_combinations[i-1])
                dlc = dlc_prev
        
        if self.grad_clipping:
            dws = self.clip_grads(dws)
            
        for i in range(self.n_layers):
            self.weights[i] -= self.lr * dws[i]
            self.biases[i] -= self.lr * dbs[i]        
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=50):
        X_train = X_train.astype(np.float32)
        y_train_onehot, self.onehot_mapping = helper.one_hot_encode(y_train)

        if X_val is not None and y_val is not None:
            X_val = X_val.astype(np.float32)
            y_val_onehot, _ = helper.one_hot_encode(y_val)

        m = X_train.shape[0]
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            X_shuffled, y_shuffled = X_train[indices], y_train_onehot[indices]
            
            batch_losses = []

            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                y_train_pred = self.f_propagation(X_batch)
                
                batch_loss = cross_entropy(y_batch, y_train_pred)
                batch_losses.append(batch_loss)

                self.b_propagation(X_batch, y_batch)
            
            mean_train_loss = np.mean(batch_losses)
            self.train_losses.append(mean_train_loss)
            
            y_train_pred = self.f_propagation(X_train)
            y_train_labels = helper.one_hot_decode(y_train_pred, self.onehot_mapping)
            train_accuracy = accuracy_score(y_train, y_train_labels)
            self.train_accuracies.append(train_accuracy)

            if X_val is not None:
                y_val_pred = self.f_propagation(X_val)
                y_val_labels = helper.one_hot_decode(y_val_pred, self.onehot_mapping)
                
                val_loss = cross_entropy(y_val_onehot, y_val_pred)
                self.val_losses.append(val_loss)
                
                val_accuracy = accuracy_score(y_val, y_val_labels)
                self.val_accuracies.append(val_accuracy)
            
                print(f"iter {epoch}: train_loss = {batch_loss:.4f}, val_loss = {val_loss:.4f}, train_acc = {train_accuracy:.4f}, val_acc = {val_accuracy:.4f}")
        
    def predict(self, X):
        y_pred = self.f_propagation(X)
        y_pred = helper.one_hot_decode(y_pred, self.onehot_mapping)
        return y_pred
       
    def clip_grads(self, grads, max_norm=1.0):
        total_norm = 0
        for grad in grads:
            total_norm += np.sum(np.square(grad))
        total_norm = np.sqrt(total_norm)
        
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for grad in grads:
                grad *= scale
        return grads
    