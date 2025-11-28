import numpy as np
import activation
from loss import cross_entropy
import helper
from sklearn.metrics import accuracy_score

np.random.seed(42)

class DNN:
    def __init__(self, epochs, layers=[3072, 512, 256, 128, 64, 5], lr=0.02, grad_clipping=True, dropout_rate=0.0, val_patience=20, use_augmenting=True, use_bn=True):
        self.lr = lr
        self.epochs = epochs
        self.n_layers = len(layers)-1
        self.grad_clipping = grad_clipping
        self.dropout_rate = dropout_rate
        self.val_patience = val_patience
        self.use_augmenting = use_augmenting
        self.use_bn = use_bn
        
        self.weights = []
        self.biases = []
        
        if use_bn:
            self.gamma = []
            self.beta = []
            self.running_means = []
            self.running_vars = []
        
        for i in range(self.n_layers):
            w = self.init_weights(layers[i], layers[i+1])
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)
            
            if self.use_bn and i < self.n_layers-1:
                gamma = np.ones((1, layers[i+1]))
                beta = np.zeros((1, layers[i+1]))
                
                self.gamma.append(gamma)
                self.beta.append(beta)
                self.running_means.append(beta)
                self.running_vars.append(gamma)
                
    
    def init_weights(self, input_size, output_size):
        var = np.sqrt(2/(input_size))
        return np.random.normal(scale=var, size=(input_size, output_size))
        
    def f_propagation(self, X, training=True):
        self.l_combinations = []
        self.activations = [X]
        self.dropout_masks = []
        self.bn_cache = [None] * (self.n_layers-1)
        for i in range(self.n_layers-1):
            lc = np.dot(self.activations[i], self.weights[i]) + self.biases[i]
            self.l_combinations.append(lc)
            
            if self.use_bn:
                bn_out, cache = self.forward_bn(
                    lc, 
                    self.gamma[i],
                    self.beta[i],
                    self.running_means[i],
                    self.running_vars[i],
                    training)
                
                self.bn_cache[i] = cache
                lc = bn_out
            
            a = activation.leaky_relu(lc)

            if training and self.dropout_rate > 0.0:
                mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(np.float32)
                a *= mask
                a /= (1 - self.dropout_rate)
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(np.ones_like)

            self.activations.append(a)

        lc = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        a = activation.softmax(lc)
        self.l_combinations.append(lc)
        self.activations.append(a)
        self.dropout_masks.append(np.ones_like(a))
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

                if self.dropout_rate > 0.0:
                    da_prev *= self.dropout_masks[i-1]
                    da_prev /= (1 - self.dropout_rate)

                da = da_prev * activation.leaky_relu_derivative(self.l_combinations[i-1])

                if self.use_bn and i - 1 < self.n_layers - 1:
                    d_bn, dgamma, dbeta = self.backward_bn(da, self.bn_cache[i-1])
                    
                    self.gamma[i-1] -= self.lr * dgamma
                    self.beta[i-1] -= self.lr * dbeta
                    
                    dlc = d_bn
                else:
                    dlc = da
                
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
        
        best_val_loss = np.inf
        val_patience_count = 0

        for epoch in range(self.epochs):
            indices = np.random.permutation(m)
            X_shuffled, y_shuffled = X_train[indices], y_train_onehot[indices]
            
            batch_losses = []
            
            for start in range(0, m, batch_size):
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                if self.use_augmenting:
                    X_batch_aug = []
                    for img in X_batch:
                        img = self.data_augment(img)
                        X_batch_aug.append(img)

                    X_batch = np.array(X_batch_aug, dtype=np.float32)                

                y_train_pred = self.f_propagation(X_batch)
                self.b_propagation(X_batch, y_batch)
                batch_loss = cross_entropy(y_batch, y_train_pred)
                batch_losses.append(batch_loss)
                
                
                
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

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    val_patience_count = 0
                    best_w = [w.copy() for w in self.weights]
                    best_b = [b.copy() for b in self.biases]
                else:
                    val_patience_count += 1

                if val_patience_count >= self.val_patience:
                    print(f"Early stopping triggered.")
                    self.weights = best_w
                    self.biases = best_b
                    break
            
    def predict(self, X_test):
        y_pred = self.f_propagation(X_test)
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

    def data_augment(self, flat_img):
        img = flat_img.reshape(32, 32, 3) * 255

        if np.random.rand() < 0.25:
            # horizontal flip
            img = img[:, ::-1, :]

        if np.random.rand() < 0.25:
            # vertical flip
            img = img[::-1, :, :]

        pad = 3
        padded = np.pad(img, ((pad, pad) ,(pad, pad), (0, 0)), mode='reflect')
        x = np.random.randint(0, 2 * pad)
        y = np.random.randint(0, 2 * pad)
        img = padded[x:x+32, y:y+32]

        brightness = 1.0 + np.random.uniform(-0.1, 0.1)
        img *= brightness

        noise = np.random.normal(0, 5, img.shape)
        img += noise

        flat_img = img.reshape(-1) / 255
        return flat_img
    
    def forward_bn(self, x, gamma, beta, running_mean, running_var, training, momentum=0.9, eps=1e-5):
        if training:
            batch_mean = np.mean(x, axis=0, keepdims=True)
            batch_var = np.var(x, axis=0, keepdims=True)
            
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)
            out = gamma * x_hat + beta
            
            running_mean[:] = momentum * running_mean + (1 - momentum) * batch_mean
            running_var[:] = momentum * running_var + (1 - momentum) * batch_var
            
            cache = (x, x_hat, batch_mean, batch_var, gamma, eps)
            return out, cache
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
            out = gamma * x_hat + beta
            return out, None
        
    def backward_bn(self, dout, cache):
        x, x_hat, mean, var, gamma, eps = cache
        m = x.shape[0]
        
        dgamma = np.sum(dout * x_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)
        
        dx_hat = dout * gamma
        dvar = np.sum(dx_hat * (x - mean) * -0.5 * (var + eps)**(-3/2), axis=0, keepdims=True)
        dmean = np.sum(dx_hat * -1 / np.sqrt(var + eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (x - mean), axis=0, keepdims=True)
        
        dx = dx_hat / np.sqrt(var + eps) + dvar * 2 * (x - mean) / m + dmean / m
        return dx, dgamma, dbeta
    