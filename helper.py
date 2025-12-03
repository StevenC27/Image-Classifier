import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def unpickle(file):
    # opens file and stores data in dict.
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        
    # stores X and y data using data and lables.
    X = np.array(dict[b'data'])
    y = np.array(dict[b'labels'])
    
    class_indices = np.where((y == 2) | (y == 3) | (y == 4) | (y == 5) | (y == 7)) # gets the indices where y is 2, 3, 4, 5 or 7.
    
    X = X[class_indices] # gets the X data for all indices in class_indices.
    y = y[class_indices] # gets the y data for all indices in class_indices.
    return X, y # returns X and y.

# unpickles training data.
def unpickle_train():
    X = np.empty((0, 3072), dtype=np.uint8) # creates an empty array for storing the training input.
    y = np.array([], dtype=np.int64) # creates an array for storing training output.
    for i in range(1, 6):
        file = f"./cifar-10-batches-py/data_batch_{i}" # gets the file name.
        
        X_i, y_i = unpickle(file) # gets the input and output data for file i.
        X_i = X_i.reshape(-1, 3072) # reshapes training images into flattened images.
        
        X = np.concatenate((X, X_i), axis=0) # concatenates X_i onto X.
        y = np.append(y, y_i) # appends y_i onto y.
    return X, y # returns X and y.

def get_train_test():
    X_train, y_train = unpickle_train() # gets the complete training input and output data.
    X_test, y_test = unpickle("./cifar-10-batches-py/test_batch") # gets the testing input and output data.
    
    X_train = X_train.astype('float32') / 255 # normalises training data to range [0, 1]
    X_test = X_test.astype('float32') / 255 # normalises testing data to range [0, 1]
    
    return X_train, y_train, X_test, y_test # returns training/testing input and output.

def one_hot_encode(y):
    unique_classes = np.unique(y) # gets the list of unique classes.
    n_classes = len(unique_classes) # gets the number of unique classes.
    
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)} # creates a mapping from classes 2, 3, 4, 5, 7 to 0, 1, 2, 3, 4
    onehot = np.zeros((y.size, n_classes)) # creates zeros of size (y.size x n_classes) and stores in onehot.
    
    # for each row in onehot, sets zero to one for the correct class.
    for i, label in enumerate(y):
        onehot[i, class_to_index[label]] = 1
        
    return onehot, class_to_index # returns onehot and its mapping.

def one_hot_decode(y_onehot, class_to_index):
    index_to_class = {v: k for k, v in class_to_index.items()} # reverses the onehot_encoding mapping.
    indices = np.argmax(y_onehot, axis=1) # gets the indices where a 1 is.
    return np.array([index_to_class[i] for i in indices]) # returns the class of the index in index_to_class.
    
def plot_loss_accuracy(train_losses, val_losses, train_accuracy, val_accuracy):
    plt.figure() # creates a new figure to plot loss and accuracy.
    plt.subplot(1, 2, 1) # activates the first subplot of a size 1x2 plot for plotting loss.
    plt.plot(train_losses, label="Train Loss") # plots the training losses.
    plt.plot(val_losses, label="Validation Loss") # plots the validation losses.
    plt.xlabel("Epochs") # sets the x label to epochs.
    plt.ylabel("Loss") # sets the y label to loss.
    plt.legend() # shows the colours of the corresponding plotting.
    
    plt.subplot(1, 2, 2) # activates the second subplot of a size 1x2 plot for plotting accuracy.
    plt.plot(train_accuracy, label="Train Accuracy") # plots the training accuracies.
    plt.plot(val_accuracy, label="Validation Accuracy") # plots the validation accuracies.
    plt.xlabel("Epochs") # sets the x label to epochs.
    plt.ylabel("Accurracy") # sets the y label to loss.
    plt.legend() # shows the colour of the corresponding plotting.
    

def plot_cm(class_labels, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred) # gets the confusion matrix for y_true and y_pred.
    cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels) # plots the confusion matrix.
    cm_plot.plot(cmap="Reds") # makes the confusion matrix red.
    