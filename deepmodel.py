import numpy as np
import activation
from loss import cross_entropy
import helper
from sklearn.metrics import accuracy_score

class DNN:
    def __init__(self,
                 layer_sizes,
                 epochs,
                 lr=0.01,
                 use_bn_layers=None,
                 use_bn=False,
                 dropout_rate=0.0,
                 early_stopping=True,
                 patience=10,
                 min_delta=1e-4,
                 weight_decay=0.0):
        self.lr = lr
        self.epochs = epochs
        self.n_layers = len(layer_sizes)-1
        self.use_bn = use_bn
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        
        self.running_mean = [np.zeros((1, layer_sizes[i+1])) for i in range(self.n_layers)]
        self.running_var = [np.ones((1, layer_sizes[i+1])) for i in range(self.n_layers)]
        self.momentum = 0.9
        
        self.use_bn_layers = use_bn_layers
        
        self.weights = []
        self.biases = []
        if use_bn:
            self.gamma = []
            self.beta = []
        
        for i in range(self.n_layers):
            w = self.init_weights(layer_sizes[i], layer_sizes[i+1])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
            if use_bn: 
                if i in self.use_bn_layers:
                    self.gamma.append(np.ones((1, layer_sizes[i+1])))
                    self.beta.append(np.zeros((1, layer_sizes[i+1])))
                else:
                    self.gamma.append(np.ones((1, layer_sizes[i+1])))
                    self.beta.append(np.zeros((1, layer_sizes[i+1])))
        
    def init_weights(self, input_size, output_size):
        var = np.sqrt(2/(input_size))
        return np.random.normal(scale=var, size=(input_size, output_size)).astype(np.float32)    
    
    def f_batch_normalisation(self, lc, gamma, beta, layer_idx, training=True, eps=1e-8):
        if training:
            mean = np.mean(lc, axis=0, keepdims=True)
            var = np.var(lc, axis=0, keepdims=True)
            
            self.running_mean[layer_idx] = (
                self.momentum * self.running_mean[layer_idx] + (1 - self.momentum) * mean
            )
            self.running_var[layer_idx] = (
                self.momentum * self.running_var[layer_idx] + (1 - self.momentum) * var
            )
        else:
            mean = self.running_mean[layer_idx]
            var = self.running_var[layer_idx]
            
        lc_norm = (lc - mean) / np.sqrt(var + eps)
        out = gamma * lc_norm + beta
        cache = (lc, lc_norm, mean, var, gamma, beta, eps)        
        return out, cache
    
    def b_batch_normalisation(self, dlc_norm, cache):
        lc, lc_norm, mean, var, gamma, beta, eps = cache
        m = lc.shape[0]
        
        dgamma = np.sum(dlc_norm * lc_norm, axis=0, keepdims=True)
        dbeta = np.sum(dlc_norm, axis=0, keepdims=True)
        
        dlc_norm_scaled = dlc_norm * gamma
        dvar = np.sum(dlc_norm_scaled * (lc - mean) * -0.5 * (var + eps) ** (-1.5), axis=0, keepdims=True)
        dmean = np.sum(dlc_norm_scaled * -1 / np.sqrt(var + eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (lc - mean), axis=0, keepdims=True)
        dlc = dlc_norm_scaled / np.sqrt(var + eps) + dvar * 2 * (lc - mean) / m + dmean / m
        
        return dlc, dgamma, dbeta
    
    def f_propagation(self, X, training=True):
        self.l_combinations = []
        self.activations = [X]
        self.batch_norm_cache = []
        self.dropout_masks = []
        for i in range(self.n_layers-1):
            lc = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            
            if self.use_bn:
                if i in self.use_bn_layers:
                    lc, cache = self.f_batch_normalisation(lc, self.gamma[i], self.beta[i], i, training=training)
                    self.batch_norm_cache.append(cache)
                else:
                    self.batch_norm_cache.append(None)
            
            a = activation.leaky_relu(lc)
            self.l_combinations.append(lc)
            
            if training and self.dropout_rate > 0.0:
                mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(np.float32)
                a *= mask
                a /= (1 - self.dropout_rate)
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(np.ones_like(a))
            
            self.activations.append(a)
            
        lc = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        a = activation.softmax(lc)
        self.l_combinations.append(lc)
        self.activations.append(a)
        self.dropout_masks.append(np.ones_like(a))
        return a

    def b2_propagation(self, X, y):
        m = X.shape[0]
        dlc = self.activations[-1] - y
        
        grads_w = [None] * self.n_layers
        grads_b = [None] * self.n_layers
        grads_gamma = [None] * self.n_layers
        grads_beta = [None] * self.n_layers
        
        for i in reversed(range(self.n_layers)):
            a_prev = self.activations[i]
            grads_w[i] = np.dot(a_prev.T, dlc) / m
            grads_b[i] = np.sum(dlc, axis=0, keepdims=True) / m
            
            if self.weight_decay > 0:
                grads_w[i] += self.weight_decay * self.weights[i]
            
            if i != 0:
                da_prev = np.dot(dlc, self.weights[i].T)
                
                if self.dropout_rate > 0.0:
                    da_prev *= self.dropout_masks[i-1]
                    da_prev /= (1 - self.dropout_rate)
                    
                dlc_prev = da_prev * activation.leaky_relu_derivative(self.l_combinations[i-1])
            
                if i-1 in self.use_bn_layers and self.batch_norm_cache[i-1] is not None:
                    dlc_prev, dgamma, dbeta = self.b_batch_normalisation(dlc_prev, self.batch_norm_cache[i-1])
                    grads_gamma[i-1] = dgamma / m
                    grads_beta[i-1] = dbeta / m
                else:
                    grads_gamma[i-1] = None
                    grads_beta[i-1] = None
                
                dlc = dlc_prev
                
        max_norm = 5
        total_norm = np.sqrt(sum(np.sum(g**2) for g in grads_w if g is not None))
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-8)
            grads_w = [g * scale if g is not None else None for g in grads_w]
            grads_b = [g * scale if g is not None else None for g in grads_b]
        
        for i in range(self.n_layers):
            self.weights[i] -= self.lr * (grads_w[i] + self.weight_decay * self.weights[i])
            self.biases[i] -= self.lr * grads_b[i]
            
            if i in self.use_bn_layers and grads_gamma[i] is not None:
                self.gamma[i] -= self.lr * grads_gamma[i]
                self.beta[i] -= self.lr * grads_beta[i]
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=128):
        X_train = X_train.astype(np.float32)
        if X_val is not None:
            X_val = X_val.astype(np.float32)
        
        y_train_onehot, self.one_hot_mapping = helper.one_hot_encode(y_train)
        y_val_onehot = helper.one_hot_encode(y_val)[0] if y_val is not None else None
        
        m = X_train.shape[0]
        best_val_accuracy = 0.0
        patience_counter = 0
        
        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            X_shuffled, y_shuffled = X_train[indices], y_train_onehot[indices]
            total_loss = 0.0
            correct_train = 0
            
            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]
                
                y_pred = self.f_propagation(X_batch, training=True)
                loss = cross_entropy(y_batch, y_pred)
                total_loss += loss
                
                self.b2_propagation(X_batch, y_batch)
                
                correct_train += np.sum(np.argmax(y_pred, 1) == np.argmax(y_batch, 1))
                
            mean_loss = total_loss / (m / batch_size)
            train_accuracy = correct_train / m
            
            if X_val is not None and y_val_onehot is not None:
                y_val_pred = self.f_propagation(X_val, training=False)
                val_loss = cross_entropy(y_val_onehot, y_val_pred)
                
                val_accuracy = np.mean(np.argmax(y_val_pred, 1) == np.argmax(y_val_onehot, 1))
            else:
                val_loss = mean_loss
                val_accuracy = train_accuracy
              
            print(f"Epoch {epoch+1}, train_loss = {mean_loss:.4f}, val_loss = {val_loss:.4f}")
            
            if val_accuracy > best_val_accuracy + self.min_delta:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                best_w = [w.copy() for w in self.weights]
                best_b = [b.copy() for b in self.biases]
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                print(f"--- Early stopping triggered at epoch {epoch+1} ---")
                self.weights = best_w
                self.biases = best_b
                break
                    
            self.lr *= 0.98

    def predict(self, X):
        y_pred = self.f_propagation(X, training=False)
        y_pred = helper.one_hot_decode(y_pred, self.one_hot_mapping)
        return y_pred
        