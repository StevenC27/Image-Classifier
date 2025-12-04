import numpy as np
from activation import leaky_relu, leaky_relu_derivative, softmax
from loss import cross_entropy, cross_entropy_derivative
import helper
from sklearn.metrics import accuracy_score

np.random.seed(42) # sets the random seed.

class MLP:
    def __init__(self, epochs, layers, lr=0.005, grad_clipping=True):
        # stores the hyper-parameters.
        self.lr = lr
        self.epochs = epochs
        self.grad_clipping = grad_clipping
        
        self.n_layers = len(layers)-1 # stores the number of layers
        
        # initialises the weights and biases to empty lists.
        self.weights = []
        self.biases = []
        
        # generates initial weights and biases for each layer.
        for i in range(self.n_layers):
            w = self.init_weights(layers[i], layers[i+1]) # creates weights of size (layers[i] x layers[i+1]) using He initialisation.
            b = np.zeros((1, layers[i+1])) # creates biases of size (1 x layers[i+1]) full of 0s.
            self.weights.append(w) # appends the w into the list of weights.
            self.biases.append(b) # appends the b into the list of biases.
        
    def init_weights(self, input_size, output_size):
        var = np.sqrt(2/(input_size)) # calculates the variance for the random weights.
        w = np.random.normal(scale=var, size=(input_size, output_size)) # generates weights using normal distribution with mean "0" and varience "var".
        return w # returns the random weights of size (input_size x output_size).
    
    def f_propagation(self, X):
        self.l_combinations = [] # initialises l_combinations to an empty list.
        self.activations = [X] # initialises activations to a list with the first elements being X.

        # loops through the layers bar the last layer.
        for i in range(self.n_layers-1):
            lc = np.dot(self.activations[i], self.weights[i]) + self.biases[i] # calculates the linear combinations at layer i.
            self.l_combinations.append(lc) # appends lc into the list of linear combinations.
            a = leaky_relu(lc) # calculates the activations of the layer i using the leaky relu.
            self.activations.append(a) # appends "a" into the list of activations.
            
        lc = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1] # calculates the linear combinations at the output layer.
        a = softmax(lc) # calculates the activations of the last layer using softmax which is the output.
        self.l_combinations.append(lc) # appends "lc" into the list of linear combinations.
        self.activations.append(a) # appends "a" into the list of activations.
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
                dlc_prev = da_prev * leaky_relu_derivative(self.l_combinations[i-1]) # 
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
        