import numpy as np
from activation import leaky_relu, leaky_relu_derivative, softmax
from loss import cross_entropy, cross_entropy_derivative
import helper
from sklearn.metrics import accuracy_score

np.random.seed(42)

class DNN:
    def __init__(self, epochs, layers, lr=0.005, grad_clipping=True, dropout_rate=0.0, dropout_layers=None, use_augmenting=True, use_bn=True):
        # stores the hyper-parameters.
        self.lr = lr
        self.epochs = epochs
        self.grad_clipping = grad_clipping
        self.dropout_rate = dropout_rate
        self.dropout_layers = dropout_layers
        self.use_augmenting = use_augmenting
        self.use_bn = use_bn

        self.n_layers = len(layers)-1 # stores the number of layers
        
        # initialises the weights and biases to empty lists.
        self.weights = []
        self.biases = []
        
        # initialises gamma, beta, running_means and running_vars if batch norm is used.
        if use_bn:
            self.gamma = []
            self.beta = []
            self.running_means = []
            self.running_vars = []
        
        # generates initial values for each layer.
        for i in range(self.n_layers):
            w = self.init_weights(layers[i], layers[i+1]) # creates weights of size (layers[i] x layers[i+1]) using He initialisation.
            b = np.zeros((1, layers[i+1])) # creates biases of size (1 x layers[i+1]) full of 0s.
            self.weights.append(w) # appends the w into the list of weights.
            self.biases.append(b) # appends the b into the list of biases.
            
            # checks if batch norm is used and that the layer is not the last layer.
            if self.use_bn and i < self.n_layers-1:
                self.gamma.append(np.ones((1, layers[i+1]))) # adds ones of size (1 x layersi[i+1]) to gamma list.
                self.beta.append(np.zeros((1, layers[i+1]))) # adds zeros of size (1 x layersi[i+1]) to beta list.
                self.running_means.append(np.zeros((1, layers[i+1]))) # adds zeros of size (1 x layersi[i+1]) to running_means list.
                self.running_vars.append(np.ones((1, layers[i+1]))) # adds ones of size (1 x layersi[i+1]) to running_vars list.
                
    def init_weights(self, input_size, output_size):
        var = np.sqrt(2/(input_size)) # calculates the variance for the random weights.
        w = np.random.normal(scale=var, size=(input_size, output_size)) # generates weights using normal distribution with mean "0" and varience "var".
        return w # returns the random weights of size (input_size x output_size).
        
    def f_propagation(self, X, training=True):
        self.l_combinations = [] # initialises l_combinations to an empty list.
        self.activations = [X] # initialises activations to a list with the first elements being X.
        self.dropout_masks = [] # initialises dropout_masks to an empty list.
        self.bn_cache = [] # initialises bn_cache to an empty list.
        self.bn_out = [] # initialises bn_out to an empty list.

        # loops through the layers bar the last layer.
        for i in range(self.n_layers-1):
            lc = np.dot(self.activations[i], self.weights[i]) + self.biases[i] # calculates the linear combinations at layer i.
            self.l_combinations.append(lc) # appends "lc" into the list of linear combinations.
            
            # checks if batch norm is used.
            if self.use_bn:
                # gets and stores bn_out and cache
                bn_out, cache = self.forward_bn(lc, self.gamma[i], self.beta[i], self.running_means[i], self.running_vars[i], training)
                
                self.bn_cache.append(cache) # appends cache into bn_cache list.
                lc_bn = bn_out # sets batch norm linear combinations to bn_out.
            else:
                lc_bn = lc # sets batch norm linear combinations to lc.
            
            self.bn_out.append(lc_bn) # appends lc_bn to bn_out.
            a = leaky_relu(lc_bn) # calculates the activations of the layer i using the leaky relu.

            # checks if training is true and dropout is active.
            if training and self.dropout_rate > 0.0 and i in self.dropout_layers:
                mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(np.float32) # creates a random mask of ones and zeros
                a *= mask # multiplies the activations by the mask, making some neurons inactive.
                a /= (1 - self.dropout_rate) # scales output by 1 - dropout_rate.
                self.dropout_masks.append(mask) # appends the mask to dropout_masks.
            else:
                # if not training or dropout is nor active then appends an array of ones with the same structure as activations to dropout_masks.
                self.dropout_masks.append(np.ones_like(a))

            self.activations.append(a) # appends activation to activations.

        lc = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1] # calculates the linear combinations at the output layer.
        a = softmax(lc) # calculates the activations of the last layer using softmax which is the output.
        self.l_combinations.append(lc) # appends "lc" into the list of linear combinations.
        self.activations.append(a) # appends "a" into the list of activations.
        self.dropout_masks.append(np.ones_like(a)) # appends ones with same dimensions as a into dropout_masks.
        return a # returns the output of forward propagation.
    
    def b_propagation(self, X, y):
        m = X.shape[0] # stores the number of images in X.
        dlc = cross_entropy_derivative(y, self.activations[-1]) # calculates the gradient of the cross entropy loss and stores it in dlc.
        dws = [None] * self.n_layers # creates an array for weight gradients of size n_layers with each element being initialised to None.
        dbs = [None] * self.n_layers # creates an array for bias gradients of size n_layers with each element being initialised to None.
        
        # loops through the layers in reverse order.
        for i in reversed(range(self.n_layers)):
            a_prev = self.activations[i] # sets a_prev to the activations[i]
            dws[i] = np.dot(a_prev.T, dlc) / m # calculates weight changes for each layer weights.
            dbs[i] = np.sum(dlc, axis=0, keepdims=True) / m # calculates bias changes for each layer biases.

            # checks if the layer is not the first layer.
            if i != 0:
                da_prev = np.dot(dlc, self.weights[i].T) # calculates how much the previous layer's activations affect the current layer.

                # checks if dropout is active and not input or output layer.
                if self.dropout_rate > 0.0 and i-1 in self.dropout_layers:
                    da_prev *= self.dropout_masks[i-1] # multiplies dropout masks to da_prev.
                    da_prev /= (1 - self.dropout_rate) # scales da_prev by (1 - dropout_rate).

                # checks if batch norm is used
                if self.use_bn:
                    # sets act_input to previous bn_out value.
                    act_input = self.bn_out[i-1]
                else:
                    # sets act_input to l_combinations[i-1]
                    act_input = self.l_combinations[i-1]

                dlc_prev = da_prev * leaky_relu_derivative(act_input) # calculates the error of the previous layer.

                # checks if batch norm is active and previous cache exists.
                if self.use_bn and self.bn_cache[i-1] is not None:
                    dlc, dgamma, dbeta = self.backward_bn(dlc_prev, self.bn_cache[i-1]) # calculates batch gradients for lc, gamma and beta.
                    self.gamma[i-1] -= self.lr * dgamma # updates gamma.
                    self.beta[i-1] -= self.lr * dbeta # updates beta.
                else:
                    # sets dlc to dlc_prev.
                    dlc = dlc_prev
                
        # checks if gradient clipping is active.
        if self.grad_clipping:
            # clips the weights gradients.
            dws = self.clip_grads(dws)
        
        # updates the weights and biases for each layer.
        for i in range(self.n_layers):
            self.weights[i] -= self.lr * dws[i]
            self.biases[i] -= self.lr * dbs[i]
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=50):
        X_train = X_train.astype(np.float32) # converts X_train to float32 type for faster, more consistent calculations.
        y_train_onehot, self.onehot_mapping = helper.one_hot_encode(y_train) # gets one hot values for y_train and stores the mapping for later reference.
        
        # checks if validation data exists.
        if X_val is not None and y_val is not None:
            X_val = X_val.astype(np.float32) # converts X_val to float32 type for faster, more consistent calculations.
            y_val_onehot, _ = helper.one_hot_encode(y_val) # gets one hot values for y_val.
            
        m = X_train.shape[0] # stores the number of images in X_train.

        # initialises losses and accuracies for train data and validation data.
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        # loops through each epoch.
        for epoch in range(self.epochs):
            indices = np.random.permutation(m) # creates a random permutation of size m.
            X_shuffled, y_shuffled = X_train[indices], y_train_onehot[indices] # applies permutation to X_train, y_train_onehot.
            
            batch_losses = [] # creates variable for batch losses and sets to empty list.
            
            # loops through range 0 to m jumping batch_size for each iteration.
            for batch_start in range(0, m, batch_size):
                X_batch = X_shuffled[batch_start : batch_start + batch_size] # stores a batch of X_shuffled from index batch_start to (batch_start + batch_size).
                y_batch = y_shuffled[batch_start : batch_start + batch_size] # stores a batch of y_shuffled from index batch_start to (batch_start + batch_size).

                # checks if data augmentation is active.
                if self.use_augmenting:
                    X_batch_aug = [] # creates an empty list for storing augmented batch data.
                    
                    # loops through each image in the batch.
                    for img in X_batch:
                        img = self.data_augment(img) # applies data augmentation to the image.
                        X_batch_aug.append(img) # appends the image into X_batch_aug.

                    X_batch = np.array(X_batch_aug, dtype=np.float32) # sets the training batch to the augmented training batch.               

                y_train_pred = self.f_propagation(X_batch) # gets and stores the prediction of the training batch.
                batch_loss = cross_entropy(y_batch, y_train_pred) # calculates and stores cross entropy loss for the batch.
                batch_losses.append(batch_loss) # appends batch_loss into the batch_losses list.
                self.b_propagation(X_batch, y_batch) # runs back propagation to update training.
                
            mean_train_loss = np.mean(batch_losses) # gets and stores the mean training loss of the batches.
            self.train_losses.append(mean_train_loss) # appends mean_train_loss to train_losses list.
            
            y_train_pred = self.f_propagation(X_train) # gets and stores the prediction of the full training data.
            y_train_labels = helper.one_hot_decode(y_train_pred, self.onehot_mapping) # gets the labels for y_train_pred using the actual training mapping.
            train_accuracy = accuracy_score(y_train, y_train_labels) # calculate and stores training accuracy.
            self.train_accuracies.append(train_accuracy) # appends train_accuracy into train_accuracies list.
            
            # checks X_val exists.
            if X_val is not None:
                y_val_pred = self.f_propagation(X_val) # gets and stores the prediction of the validation.
                y_val_labels = helper.one_hot_decode(y_val_pred, self.onehot_mapping) # gets the labels for y_val_pred using the actual training mapping.
                
                val_loss = cross_entropy(y_val_onehot, y_val_pred) # calculates and stores cross entropy loss for the validation.
                self.val_losses.append(val_loss) # appends val_loss into the val_losses list.
                
                val_accuracy = accuracy_score(y_val, y_val_labels) # calculate and stores validation accuracy.
                self.val_accuracies.append(val_accuracy) # appends val_accuracy into val_accuracies list.
                
                # prints the current epoch, train_loss, val_loss, train_accuracy, val_accuracy.
                print(f"Epoch {epoch+1}: train_loss = {mean_train_loss:.4f}, val_loss = {val_loss:.4f}, train_acc = {train_accuracy:.4f}, val_acc = {val_accuracy:.4f}")
            
    def predict(self, X_test):
        y_pred = self.f_propagation(X_test) # gets the predictions and stores in y_pred.
        y_pred = helper.one_hot_decode(y_pred, self.onehot_mapping) # applies one_hot_decoding, updating y_pred to 1s and 0s.
        return y_pred # return predictions.
    
    def clip_grads(self, grads, max_norm=1.0):
        total_norm = np.sqrt(sum(np.sum(grad**2) for grad in grads)) # calculates the total norm.
        
        # checks if total_norm is larger than max_norm
        if total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6) # calculates scale for normalising.
            grads = [grad * scale for grad in grads] # multiplies each gradient by the scale.
        return grads # returns gradients.

    def data_augment(self, flat_img, pad=3):
        img = flat_img.reshape(32, 32, 3) * 255 # reshapes the flattened images into their original shape and unnormalises data.

        # randomly flips the image horizontally with a 25% chance.
        if np.random.rand() < 0.25:
            img = img[:, ::-1, :]

        # randomly flips the image vertically with a 25% chance.
        if np.random.rand() < 0.25:
            img = img[::-1, :, :]

        padded = np.pad(img, ((pad, pad) ,(pad, pad), (0, 0)), mode='reflect') # expands the images to a 40x40x3 image with reflect mode.
        x = np.random.randint(0, 2 * pad) # gets a random integer value from 0 to 2*pad and stores in x.
        y = np.random.randint(0, 2 * pad) # gets a random integer value from 0 to 2*pad and stores in x.
        img = padded[x:x+32, y:y+32] # crops the 40x40x3 image into 32x32x3 with random x and y start points.

        brightness = 1.0 + np.random.uniform(-0.1, 0.1) # gets a random brightness multiplies from range [0.9, 1.1].
        img *= brightness # applies the brightness multiplier to the image pixels.

        noise = np.random.normal(0, 5, img.shape) # create random point noises with the same shape as img.
        img += noise # adds noise to img.

        flat_img = img.reshape(-1) / 255 # flattens image again and normalises.
        return flat_img # returns flat_img.
    
    def forward_bn(self, lc, gamma, beta, running_mean, running_var, training, momentum=0.9, eps=1e-5):
        # checks if training is true.
        if training:
            # calculates batch_mean and batch_var
            batch_mean = np.mean(lc, axis=0, keepdims=True)
            batch_var = np.var(lc, axis=0, keepdims=True)
            
            lc_hat = (lc - batch_mean) / np.sqrt(batch_var + eps) # calculates the batch normalised linear combinations.
            out = gamma * lc_hat + beta # calculates the final output of batch normalisation.
            
            running_mean[:] = momentum * running_mean + (1 - momentum) * batch_mean # updates the running mean.
            running_var[:] = momentum * running_var + (1 - momentum) * batch_var # updates the running varience.
            
            cache = (lc, lc_hat, batch_mean, batch_var, gamma, eps) # stores all useful info in a cache for backward batch norm.
            return out, cache # returns the output and the cache.
        else:
            # if not training then a prediction is being made.
            lc_hat = (lc - running_mean) / np.sqrt(running_var + eps) # calculates the batch normalised linear combinations with the running mean and var.
            out = gamma * lc_hat + beta # calculates the final output of batch normalisation.
            return out, None # returns only the output.
        
    def backward_bn(self, dout, cache):
        lc, lc_hat, mean, var, gamma, eps = cache # unpacks the cache.
        m = lc.shape[0] # gets and stores the number of samples.
        
        dgamma = np.sum(dout * lc_hat, axis=0, keepdims=True) # calculates gradient of gamma.
        dbeta = np.sum(dout, axis=0, keepdims=True) # calculates gradient of beta.
        
        dlc_hat = dout * gamma # calculates gradient of lc_hat.
        dvar = np.sum(dlc_hat * (lc - mean) * -0.5 * (var + eps)**(-1.5), axis=0, keepdims=True) # calculates the gradient of the batch var.
        dmean = np.sum(dlc_hat * -1 / np.sqrt(var + eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (lc - mean), axis=0, keepdims=True) # calculates the gradient of the batch mean.
        
        dlc = dlc_hat / np.sqrt(var + eps) + dvar * 2 * (lc - mean) / m + dmean / m # calculates the gradient of the linear combinations.
        return dlc, dgamma, dbeta # returns dlc, dgamma and dbeta.
    