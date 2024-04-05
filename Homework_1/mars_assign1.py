import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def readPickle(file):
	""" Copied from the dataset website """
	# import pickle5 as pickle 
	with open(file, 'rb') as f:
		dict=pickle.load(f, encoding='bytes')
	f.close()
	return dict

def initialize_parameters(X, Y):
  d = X.shape[1]
  k = Y.shape[1]

  W = np.random.normal(loc=0, scale=0,size=(k, d))
  b = np.random.normal(loc=0, scale=0,size=(k, 1))
    
  print(f"INFO: W&B init: W={W.shape}, b={b.shape}")
  return W, b


def vis_weights(W, label_names, casename='None'):

  fig, ax = plt.subplots(nrows=1, ncols= len(label_names), figsize=(15,1.5))

  for k, label in enumerate(label_names):
    Wimg = W[k, :]

    ax[k].imshow(Wimg.reshape(32,32,3,order="F"))
    ax[k].axis("off")

    if casename != 'None':
       ax[k].set_title(label_names[k])
       ax[k].axis("off")
       plt.savefig(f"./04_trained_weights-{casename}.png")

    plt.savefig(path + 'vis_weights_size.png')

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
    plt.savefig(path + f'weights_f{i}_1_batch_size.png')
    plt.show()

def montage_seq(W,c):
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
    plt.savefig(path + f'weights_f{c}_1_batch_size.png')
    plt.show()
  
def softmax(s):

  return np.exp(s)/np.sum(np.exp(s), axis=0)


def EvaluateClassifier(X, W, b):
    #print('SIZES eval',W.shape,X.shape,b.shape)
    s = W @ X.T + b
    return softmax(s)

def ComputeCost_s(X,Y,W,b,lamda,return_loss = False):
	"""
	Compute the cost function: c = loss + regularisation 

	Args: 
		X	: [d,n] input 
		Y	: [K,n] One-Hot Ground Truth 
		W 	: [K,d] Weight 
		b	: [d,1] bias 
		lamb: (float) a regularisation term for the weight
	
	Return:
		J	: (float) A scalar of the cost function 
	"""
	
	# Part 1: compute the loss:
	## 1 Compute prediction:
	P = EvaluateClassifier(X,W,b)
	## Cross-entropy loss
	# Clip the value to avoid ZERO in log
	P = np.clip(P,1e-16,1-1e-16)
	l_cross =  -np.mean(np.sum(Y*np.log(P.T),axis=0))
	# Part 2: Compute the regularisation 
	reg = lamda * np.sum(W**2)
	# Assemble the components
	J = l_cross + reg
	
	del P, W

	return J, l_cross

def ComputeCost(X, Y, W, b, lamda):

  p = EvaluateClassifier(X, W, b)
  p = np.clip(p, 1e-15, 1 - 1e-15)

  loss_cross = -np.mean(np.sum(Y*np.log(p.T), axis = 1))

  reg = lamda*np.sum(W**2)

  Total = loss_cross + reg
  #print(f"total :{Total},reg: {reg}")
  return Total, loss_cross

def ComputeAccuracy(X, Y, W, b):

  p = EvaluateClassifier(X, W, b)
  p = np.argmax(p,axis=0)
  true_pos = np.sum(p==Y)   # what if p is higher value but not equal to one
  acc = true_pos/Y.shape[0]

  return acc

def ComputeGradients(X, Y, P, W, lamda):
    n = X.shape[0]
    k = Y.shape[1]
    G = -(Y.T-P)
    #print(f"{G.shape}, {X.shape}")
    grad_W = (G@X)/n+2*lamda*W
    grad_b = (G@np.ones(shape=(n,1))/n).reshape(k,1)

    return grad_W, grad_b


def ComputeGradsNum(X, Y, W, b, lamda, h=0.00001):
    grad_W = np.zeros(shape=W.shape)
    grad_b = np.zeros(shape=b.shape)
    c, c_cross = ComputeCost(X, Y, W, b, lamda)

    for i in range(b.shape[0]):
        b_copy = b.copy()
        b_copy[i,0] = b_copy[i,0]+h
        c2,c2_cross = ComputeCost(X, Y, W, b_copy, lamda)
        #print('SHAPES',c2.size,c.shapes)
        grad_b[i,0] = (c2-c)/h

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_copy = W.copy()
            W_copy[i,j] = W_copy[i,j]+h
            c2,c2_cross = ComputeCost(X, Y, W_copy, b, lamda)
            grad_W[i,j] = (c2-c)/h

    del W_copy, b_copy

    return grad_W, grad_b

def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
	""" Converted from matlab code """
	no 	= 	W.shape[0]
	d 	= 	X.shape[0]

	grad_W = np.zeros(W.shape);
	grad_b = np.zeros((no, 1));
	
	for i in range(len(b)):
		b_try = np.array(b)
		b_try[i] -= h
		c1 = ComputeCost_s(X, Y, W, b_try, lamda)

		b_try = np.array(b)
		b_try[i] += h
		c2 = ComputeCost_s(X, Y, W, b_try, lamda)

		grad_b[i] = (c2-c1) / (2*h)

	for i in range(W.shape[0]):
		for j in range(W.shape[1]):
			W_try = np.array(W)
			W_try[i,j] -= h
			c1 = ComputeCost_s(X, Y, W_try, b, lamda)

			W_try = np.array(W)
			W_try[i,j] += h
			c2 = ComputeCost_s(X, Y, W_try, b, lamda)

			grad_W[i,j] = (c2-c1) / (2*h)

	return [grad_W, grad_b]


def LoadBatch(filename):
    data = unpickle('/home/mech/sanchis/sanchis/PHD_COURSES/Deep_learning/Assign_1/DirName/Datasets/cifar-10-batches-py/' + filename)
    # print(data.keys())
    X = np.array(data[b'data']).astype(np.float64)
    y = np.array(data[b'labels']).astype(np.float64).flatten()
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(np.array(y).reshape(-1, 1))

    return X, Y, y



def preprocess_data(X_trn, X_tst, X_val):

  mu    = np.mean(X_trn, axis=0).reshape(1,-1)
  sigma = np.std(X_trn, axis=0).reshape(1,-1)

  X_trn = (X_trn - mu)/sigma
  X_tst = (X_tst - mu)/sigma
  X_val = (X_val - mu)/sigma

  return X_trn, X_tst, X_val

def split_into_batches(X,Y, y, batch_size):
    # Get the number of samples in the data
    num_samples = len(X)

    # Shuffle the indices of the data
    shuffled_indices = np.random.permutation(num_samples)

    # Split the shuffled indices into batches

    batches = { 'X':[], 'Y':[], 'y':[] }
    batches['X'] = []
    batches['Y'] = []
    batches['y'] = []

    for i in range(0, num_samples, batch_size):

        batch_indices = shuffled_indices[i:i+batch_size]
        batches['X'].append(X[batch_indices])
        batches['Y'].append(Y[batch_indices])
        batches['y'].append(y[batch_indices])

    return batches

def MiniBatchGD(X, Y, y, hyper, lamda, X_val, Y_val, y_val):

    n = X.shape[0]
    eta = hyper['eta']
    batch_size = hyper['bs']
    epochs = hyper['epochs']

    W, b = initialize_parameters(X, Y)

    # copy weight and bias
    W = W.copy()
    b = b.copy()

    loss_train = []
    loss_val = []
    accuracy = []

    batches = split_into_batches(X, Y, y, batch_size)

    counter = 0
    for epoch in range(epochs):

        for j in range(len(batches['X'])):

            X_batch = batches['X'][j]
            Y_batch = batches['Y'][j]
            y_batch = batches['y'][j]

            P_batch = EvaluateClassifier(X_batch, W, b)
            grad_W, grad_b = ComputeGradients(X_batch, Y_batch, P_batch, W, lamda)
            W += -eta * grad_W
            b += -eta * grad_b

        loss_train.append(ComputeCost(X, Y, W, b, lamda)[0])
        loss_val.append(ComputeCost(X_val, Y_val, W, b, lamda)[0])
        accuracy.append(ComputeAccuracy(X, y, W, b))

        print(f"#Epoch:{counter}  training loss = {loss_train[epoch]}  Validation loss = {loss_val[epoch]}  Accuracy = {accuracy[epoch]}")
        counter += 1

    return W, b, loss_train, loss_val, accuracy


def plot(loss_train, loss_val, color, casename):
    
    fig, ax = plt.subplots(nrows =1 , ncols =1)
    epochs = range(1, len(loss_train[0]) + 1)

    for i in range(len(loss_train)):

      ax.plot(epochs, loss_train[i], f'{color[i]}-', label= f' {casename[i]}')
      ax.plot(epochs, loss_val[i], f'{color[i]}--', label='Val loss')

    ax.set_title('Training and Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    return fig, ax
def plot_acc(acc, color, casename):
    
    fig, ax = plt.subplots(nrows =1 , ncols =1)
    epochs = range(1, len(acc[0]) + 1)

    for i in range(len(loss_train)):

      ax.plot(epochs, acc[i], f'{color[i]}-', label= f' {casename[i]}')

    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()

    return fig, ax
################## MAIN CODE TEST #######################
print('-------------START TESTING-------')
"""Test the functions for the assignments """

# Load CIFAR-10 dataset

data_batch1 = "data_batch_1"
data_batch2 = "data_batch_2"
test_batch  = "test_batch"

X_trn, Y_trn, y_trn = LoadBatch(data_batch1)
X_val, Y_val, y_val = LoadBatch(data_batch2)
X_tst, Y_tst, y_tst = LoadBatch(test_batch)

X_trn, X_tst, X_val = preprocess_data(X_trn, X_tst, X_val)

path = "/home/mech/sanchis/sanchis/PHD_COURSES/Deep_learning/Assign_1/DirName/Datasets/cifar-10-batches-py/"
batches = unpickle(path + "batches.meta")
label_names = [label_name.decode('utf-8') for label_name in batches[b'label_names']]

_, axes = plt.subplots(nrows=10, ncols=5, figsize=(5,10))

for label_index, label_name in enumerate(label_names):
    images = X_tst[y_tst == label_index,:][0:5, :]
    for image_index, image in enumerate(images):
        axes[label_index][image_index].imshow((image.reshape(3,32,32).T))
        axes[label_index][image_index].tick_params(bottom=False, top=False, left=False, right=False,
                                                   labelbottom=False, labelleft=False)
    axes[label_index][0].set_ylabel(label_name, labelpad=50, rotation=0, size='large')
plt.savefig(path + '/sample_images.png', bbox_inches="tight")


W, b = initialize_parameters(X_trn, Y_trn)
montage(X_trn)

X_small = X_tst[:, :40]
Y_small = Y_tst[:, :]

W_small, b = initialize_parameters(X_small, Y_small)

lamda = 0

P_small = EvaluateClassifier(X_small,W_small , b)

anl_grad_W, anl_grad_b = ComputeGradients(X_small, Y_small, P_small, W_small, lamda)

num_grad_W, num_grad_b =  ComputeGradsNum(X_small, Y_small, W_small, b, lamda, h=1e-6)

#num_grad_W_s, num_grad_b_s =  ComputeGradsNumSlow(X_small, Y_small, P_small, W_small, b, lamda, h=1e-6)
print('-------------FINISH TESTING-------')

#%%
print('------------- ERRORS -------------')
grad_W_abs_diff = np.abs(num_grad_W-anl_grad_W)
grad_b_abs_diff = np.abs(num_grad_b-anl_grad_b)
print(grad_W_abs_diff)


print('For weights: '+str(np.mean(grad_W_abs_diff<1e-6)*100)+"% of absolute errors below 1e-6")
print('For bias: '+str(np.mean(grad_b_abs_diff<1e-6)*100)+"% of absolute errors below 1e-6")
print('For weights the maximum absolute error is '+str((grad_W_abs_diff).max()))
print('For bias the maximum absolute error is '+str((grad_b_abs_diff).max()), "\n")


# Relative error between numerically and analytically computed gradient.
grad_W_abs_sum = np.maximum(np.abs(num_grad_W)+np.abs(anl_grad_W), 0.00000001)
grad_b_abs_sum = np.maximum(np.abs(num_grad_b)+np.abs(anl_grad_b), 0.00000001)
print('For weights: '+str(np.mean(grad_W_abs_diff/grad_W_abs_sum<1e-6)*100)+
      "% of relative errors below 1e-6")
print('For bias: '+str(np.mean(grad_b_abs_diff/grad_b_abs_sum<1e-6)*100)+
      "% of relative errors below 1e-6")
print('For weights the maximum relative error is '+str((grad_W_abs_diff/grad_W_abs_sum).max()))
print('For bias the maximum relative error is '+str((grad_b_abs_diff/grad_b_abs_sum).max()))
#%%

######## TRAINING PERIOD ###########
print('-------------START TRAINING-------')
hyper = {'bs':64, 'eta':0.001, 'epochs':50}
lamda = 0.0
W_s1p1, b_s1p1, loss_train_s1p1, loss_val_s1p1, acc_s1p1  = MiniBatchGD(X_trn, Y_trn, y_trn, hyper, lamda, X_val, Y_val, y_val )
hyper = {'bs':128, 'eta':0.001, 'epochs':50}
lamda = 0.0
W_s1p2, b_s1p2, loss_train_s1p2, loss_val_s1p2, acc_s1p2 = MiniBatchGD(X_trn, Y_trn, y_trn, hyper, lamda, X_val, Y_val, y_val )
hyper = {'bs':256, 'eta':0.001, 'epochs':50}
lamda = 0.0
W_s1p3, b_s1p3, loss_train_s1p3, loss_val_s1p3, acc_s2p1, acc_s2p2 = MiniBatchGD(X_trn, Y_trn, y_trn, hyper, lamda, X_val, Y_val, y_val )

color = ['r', 'b', 'g']
casename=['bs64', 'bs128', 'bs256']
loss_train = [loss_train_s1p1, loss_train_s1p2, loss_train_s1p3]
loss_val = [loss_val_s1p1, loss_val_s1p2, loss_val_s1p3 ]
W_trained = [W_s1p1, W_s1p2, W_s1p3]

print('-------------- PLOTTING -------------')
fig, ax = plot(loss_train, loss_val, color, casename)
plt.savefig(path + '02_batch_size.png')

for i in range(len(W_trained)):
  montage(W_trained[i])
  
print('-------------FINISH TRAINING-------')
#%%

print('------------ ASSIGNMENT STAGE -------------')
lamda = 0.0
hyper = {'bs':100, 'eta':0.1, 'epochs':40}
W_s1p1, b_s1p1, loss_train_s1p1, loss_val_s1p1, acc_s1p1 = MiniBatchGD(X_trn, Y_trn, y_trn, hyper, lamda, X_val, Y_val, y_val )
print('---------- FIRST-----------------')
hyper = {'bs':100, 'eta':0.001, 'epochs':40}
W_s1p2, b_s1p2, loss_train_s1p2, loss_val_s1p2, acc_s1p2 = MiniBatchGD(X_trn, Y_trn, y_trn, hyper, lamda, X_val, Y_val, y_val )
print('---------- SECOND -----------------')

lamda = 0.1
hyper = {'bs':100, 'eta':0.001, 'epochs':40}
W_s2p1, b_s2p1, loss_train_s2p1, loss_val_s2p1, acc_s2p1 = MiniBatchGD(X_trn, Y_trn, y_trn, hyper, lamda, X_val, Y_val, y_val )
print('---------- THIRD -----------------')
lamda = 1
hyper = {'bs':100, 'eta':0.001, 'epochs':40}
W_s2p2, b_s2p2, loss_train_s2p2, loss_val_s2p2, acc_s2p2 = MiniBatchGD(X_trn, Y_trn, y_trn, hyper, lamda, X_val, Y_val, y_val )
print('---------- FOURTH -----------------')

colours = ['m', 'y','g']
casename=['0.001', '0.001', '0.001']
casename_t=['0.1','0.001', '0.001', '0.001']
casename_1=['0.1']
colours_1 = ['r']
loss_train = [ loss_train_s1p2, loss_train_s2p1,loss_train_s2p2]
loss_val = [loss_val_s1p2, loss_val_s2p1,loss_val_s2p2 ]
acc_1 =[acc_s1p1]
acc = [acc_s1p2,acc_s2p1, acc_s2p2]
W_trained = [W_s1p1,W_s1p2, W_s2p2, W_s1p3]
loss_1 = [loss_train_s1p1]
val_1 = [loss_val_s1p1]

fig, ax = plot(loss_1, val_1, colours_1, casename_1)
plt.savefig(path + 'loss_1_batch_size.png')
fig, ax = plot(loss_train, loss_val, colours, casename)
plt.savefig(path + 'loss_3_batch_size.png')
fig , ax = plot_acc(acc, colours, casename)
plt.savefig(path + 'val_3_batch_size.png')
fig , ax = plot_acc(acc_1, colours_1, casename_1)
plt.savefig(path + 'val_1_batch_size.png')

#%%

for i in range(len(W_trained)):
    montage_seq(W_trained[i],i) 
  
#%%