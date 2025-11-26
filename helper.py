import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def unpickle_train():
    X = np.empty((0, 3072), dtype=np.uint8)
    y = np.array([], dtype=np.int64)
    for i in range(1, 6):
        file = f"./cifar-10-batches-py/data_batch_{i}"
        
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            
        y_i = np.array(dict[b'labels'])
        print(y_i)
        X_i = np.array(dict[b'data'])
        
        class_indices = np.where((y_i == 2) | (y_i == 3) | (y_i == 4) | (y_i == 5) | (y_i == 7))
        
        y_i = y_i[class_indices]
        X_i = X_i[class_indices]
        
        X_i = X_i.reshape(-1, 3072)
        
        X = np.concatenate((X, X_i), axis=0)
        y = np.append(y, y_i)
        
    X = X.astype('float32') / 255
    return X, y

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
        
    X = np.array(dict[b'data'])
    y = np.array(dict[b'labels'])
    
    class_indices = np.where((y == 2) | (y == 3) | (y == 4) | (y == 5) | (y == 7))
    
    X = X[class_indices]
    y = y[class_indices]
        
    X = X.astype('float32') / 255
    return X, y

def get_test_train():
    """
    X_train = []
    y_train = []
    for i in range(1, 6):
        file = f"./cifar-10-batches-py/data_batch_{i}"
        X_train_i, y_train_i = unpickle(file)
        X_train.append(X_train_i)
        y_train.append(y_train_i)
        """
    X_train, y_train = unpickle_train()
        
    X_test, y_test = unpickle("./cifar-10-batches-py/test_batch")
    return X_train, y_train, X_test, y_test

def one_hot_encode(y):
    unique_classes = np.unique(y)
    n_classes = len(unique_classes)
    
    class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
    onehot = np.zeros((y.size, n_classes))
    for i, label in enumerate(y):
        onehot[i, class_to_index[label]] = 1
        
    return onehot, class_to_index

def one_hot_decode(y_onehot, class_to_index):
    index_to_class = {v: k for k, v in class_to_index.items()}
    indices = np.argmax(y_onehot, axis=1)
    return np.array([index_to_class[i] for i in indices])
    
def plot_loss_accuracy(train_losses, val_losses, train_accuracy, val_accuracy):
    plt.figure()
    plt.subplot(1, 2, 1)    
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accurracy")
    plt.legend()
    

def plot_cm(class_labels, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_plot = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    cm_plot.plot()
    