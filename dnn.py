import numpy as np
import activation
from loss import cross_entropy
import helper
from sklearn.metrics import accuracy_score

np.random.seed(42)

class DNN:
    def __init__(self, epochs, layers=[3072, 512, 256, 128, 64, 5], lr=0.005, grad_clipping=True, dropout_rate=0.0, dropout_layers=None, val_patience=10, use_augmenting=True, use_bn=True, weight_decay=0.0):
        self.lr = lr
        self.epochs = epochs
        self.n_layers = len(layers)-1
        self.grad_clipping = grad_clipping
        self.dropout_rate = dropout_rate
        self.dropout_layers = dropout_layers
        self.val_patience = val_patience
        self.use_augmenting = use_augmenting
        self.use_bn = use_bn
        self.weight_decay = weight_decay
        
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
                self.gamma.append(np.ones((1, layers[i+1])))
                self.beta.append(np.zeros((1, layers[i+1])))
                self.running_means.append(np.zeros((1, layers[i+1])))
                self.running_vars.append(np.ones((1, layers[i+1])))
                
    
    def init_weights(self, input_size, output_size):
        var = np.sqrt(2/(input_size))
        return np.random.normal(scale=var, size=(input_size, output_size))
        
    def f_propagation(self, X, training=True):
        self.l_combinations = []
        self.activations = [X]
        self.dropout_masks = []
        self.bn_cache = [None] * self.n_layers
        self.bn_out = []
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
                lc_bn = bn_out
            else:
                lc_bn = lc
            
            self.bn_out.append(lc_bn)
            a = activation.leaky_relu(lc_bn)

            if training and self.dropout_rate > 0.0 and i in self.dropout_layers:
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

                if self.dropout_rate > 0.0 and i-1 in self.dropout_layers:
                    da_prev *= self.dropout_masks[i-1]
                    da_prev /= (1 - self.dropout_rate)

                if self.use_bn:
                    act_input = self.bn_out[i-1]
                else:
                    act_input = self.l_combinations[i-1]

                dlc_prev = da_prev * activation.leaky_relu_derivative(act_input)

                if self.use_bn and self.bn_cache[i-1] is not None:
                    dlc, dgamma, dbeta = self.backward_bn(dlc_prev, self.bn_cache[i-1])
                    self.gamma[i-1] -= self.lr * dgamma
                    self.beta[i-1] -= self.lr * dbeta
                else:
                    dlc = dlc_prev
                
        if self.grad_clipping:
            dws = self.clip_grads(dws)

        for i in range(self.n_layers):
            if self.weight_decay > 0:
                self.weights[i] -= self.lr * (dws[i] + self.weight_decay * self.weights[i])
            else:
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
    
    def forward_bn(self, lc, gamma, beta, running_mean, running_var, training, momentum=0.9, eps=1e-5):
        if training:
            batch_mean = np.mean(lc, axis=0, keepdims=True)
            batch_var = np.var(lc, axis=0, keepdims=True)
            
            lc_hat = (lc - batch_mean) / np.sqrt(batch_var + eps)
            out = gamma * lc_hat + beta
            
            running_mean[:] = momentum * running_mean + (1 - momentum) * batch_mean
            running_var[:] = momentum * running_var + (1 - momentum) * batch_var
            
            cache = (lc, lc_hat, batch_mean, batch_var, gamma, eps)
            return out, cache
        else:
            lc_hat = (lc - running_mean) / np.sqrt(running_var + eps)
            out = gamma * lc_hat + beta
            return out, None
        
    def backward_bn(self, dout, cache):
        lc, lc_hat, mean, var, gamma, eps = cache
        m = lc.shape[0]
        
        dgamma = np.sum(dout * lc_hat, axis=0, keepdims=True)
        dbeta = np.sum(dout, axis=0, keepdims=True)
        
        dlc_hat = dout * gamma
        dvar = np.sum(dlc_hat * (lc - mean) * -0.5 * (var + eps)**(-1.5), axis=0, keepdims=True)
        dmean = np.sum(dlc_hat * -1 / np.sqrt(var + eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (lc - mean), axis=0, keepdims=True)
        
        dlc = dlc_hat / np.sqrt(var + eps) + dvar * 2 * (lc - mean) / m + dmean / m
        return dlc, dgamma, dbeta
    