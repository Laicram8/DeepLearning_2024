#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 07:26:03 2024

@author: sanchis
"""

import numpy as np
import h5py
import pickle

def montage(W):
	""" Display the image for each label in W """
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots(2,5)
	for i in range(2):
		for j in range(5):
			im  = W[i*5+j,:].reshape(32,32,3, order='F')
			sim = (im-np.min(im[:]))/(np.max(im[:])-np.min(im[:]))
			sim = sim.transpose(1,0,2)
			ax[i][j].imshow(sim, interpolation='nearest')
			ax[i][j].set_title("y="+str(5*i+j))
			ax[i][j].axis('off')
	plt.show()

def LoadBatch(filename):
    ##Copied from the dataset website.
    with open(filename, 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
    
    X = data_dict[b'data'] / 255.0  # Normalize pixel values to [0, 1]
    Y = np.eye(10)[data_dict[b'labels']].T  # One-hot encoding of labels
    y = np.array(data_dict[b'labels']) # Convert labels to 1-10 range

    return X, Y, y, data_dict
"""
def LoadBatch(filename):
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return X, Y, y
"""
def SaveToHDF5(X, Y, y, output_filename):
    # Save to HDF5
    with h5py.File(output_filename, 'w') as f:
        f.create_dataset('X', data=X)
        f.create_dataset('Y', data=Y)
        f.create_dataset('labels', data=y)
        
def means(X,Y):
    mean_X = np.mean(X, axis=1, keepdims=True)
    std_X = np.std(X, axis=1, keepdims=True)
    mean_Y = np.mean(Y, axis=1, keepdims=True)
    std_Y = np.std(Y, axis=1, keepdims=True)
    
    return  mean_X, std_X, mean_Y, std_Y
    
    return mean_X, std_X, mean_Y, std_Y 
def preprocess_data(X,Y, mean_X, std_X,mean_Y, std_Y):
    X = X - mean_X
    X = X / std_X
    Y = Y - mean_Y.reshape(-1, 1)
    Y = Y / std_Y.reshape(-1, 1)
    return X, Y

def initialize_parameters(K, d):
    W = np.random.randn(K, d) * 0.01  # Initialize W with Gaussian random values
    b = np.random.randn(K, 1) * 0.01   # Initialize b with Gaussian random values
    return W, b

def EvaluateClassifier(X, W, b):
    
    def softmax(x):
        """ Standard definition of the softmax function """
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    """
    Evaluate the network function on multiple images.
    
    Args:
    - X: Input data matrix where each column corresponds to an image. It has size d x n.
    - W: Weight matrix of the network. It has size K x d.
    - b: Bias vector of the network. It has size K x 1.
    
    Returns:
    - P: Matrix containing the probability for each label for the images in X. It has size K x n.
    """
    s = np.matmul(W, X) + b
    print(s.shape)
    P = softmax(s)
    return P


def ComputeCost(X, Y, W, b, lambda_):
    """
    Compute the cost function for a set of images.

    Args:
    - X: Input data matrix where each column corresponds to an image. It has size d x n.
    - Y: Ground truth labels matrix. Each column is either a one-hot vector of size K or a vector of labels (1 x n).
    - W: Weight matrix of the network. It has size K x d.
    - b: Bias vector of the network. It has size K x 1.
    - lambda_: Regularization parameter.

    Returns:
    - J: Scalar corresponding to the sum of the loss of the network's predictions for the images in X
         relative to the ground truth labels and the regularization term on W.
    """
    n = X.shape[1]  # Number of images
    P = EvaluateClassifier(X, W, b)

    # Ensure Y is one-hot encoded
    if Y.ndim == 1:
        Y_one_hot = np.eye(W.shape[0])[Y].T  # Convert labels vector to one-hot matrix
    else:
        Y_one_hot = Y.T
        
    # Compute cross-entropy loss
    cross_entropy_loss =-np.matmul(Y_one_hot,np.log(P))
    data_loss = np.sum(cross_entropy_loss) / n
    regularization_term = 0.5 * lambda_ * np.sum(W**2)
    
    C = data_loss + regularization_term
    return C

def ComputeAccuracy(X, y, W, b):
    """
    Compute the accuracy of the network's predictions.

    Args:
    - X: Input data matrix where each column corresponds to an image. It has size d x n.
    - y: Ground truth labels vector of length n.
    - W: Weight matrix of the network. It has size K x d.
    - b: Bias vector of the network. It has size K x 1.

    Returns:
    - acc: Scalar value containing the accuracy of the network's predictions.
    """
    n = X.shape[1]  # Number of images
    P = EvaluateClassifier(X, W, b)
    
    # Get the index of the highest probability prediction for each image
    print(P.shape)
    predictions = np.argmax(P, axis=0)
    print(predictions.shape)
    
    # Compare predictions with ground truth labels
    correct_predictions = np.mean(predictions == y)
    
    # Compute accuracy
    acc = correct_predictions / n
    
    return acc





################## MAIN CODE #######################

# Example usage:
X, Y, y, data_t = LoadBatch('Datasets/cifar-10-batches-py/data_batch_1')
Xv, Yv, yv, data_v = LoadBatch('Datasets/cifar-10-batches-py/data_batch_2')
Xt, Yt, yt,data_t = LoadBatch('Datasets/cifar-10-batches-py/test_batch')
print('READING BATCHES')
print("Shape of K:", Y[0].size)
print("Shape of nx:", X[1].size)
print("Shape of ny:", Y[1].size)
print("Shape of d:", X[0].size)

mean_X, std_X, mean_Y, std_Y = means(X,Y) 
train_X,train_Y = preprocess_data(X,Y, mean_X, std_X,mean_Y, std_Y)
Val_X,Val_Y = preprocess_data(Xv,Yv, mean_X, std_X,mean_Y, std_Y)
Test_X,Test_Y = preprocess_data(Xt,Yt, mean_X, std_X,mean_Y, std_Y)
montage(train_X)

K = np.shape(train_Y)[0]  # Number of classes
d = np.shape(train_X)[0]  # Dimensionality of each image
W, b = initialize_parameters(K, d)
print("Shape of K:", np.shape(train_Y)[0])
print("Shape of nx:", np.shape(train_X)[1])
print("Shape of ny:",np.shape(train_Y)[1])
print("Shape of d:", np.shape(train_X)[0])
print("Shape of W:", W.shape)
print("Shape of b:", b.shape)

#P = EvaluateClassifier(train_X, W, b)
#print("Shape of P:", P.shape)
Cost = ComputeCost(train_X, train_Y, W, b,2)
accuracy = ComputeAccuracy(train_X, train_Y, W, b)
print(Cost,accuracy)
#SaveToHDF5(X, Y, y, 'cifar10_data.h5')








# Now, let's access and print the labels dataset
with h5py.File('cifar10_data.h5', 'r') as f:
    labels = f['labels'][:]
    print(labels)



#SaveToHDF5(data_dict, 'cifar10_data.h5')



