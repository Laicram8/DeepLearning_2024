import sys
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
sys.path.insert(0,'/home/mech/sanchis/sanchis/PHD_COURSES/Deep_learning/Assign_1/DirName/Assignment1/')
print("Inserted path:", sys.path[0])  # Print the inserted path for debugging

#from Assignment1.mars_assign1 import *

#############3 DATA ################
def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    return data['data']


def initialize_parameters(X, Y):
    # Initialize weights
    d = X.shape[1]
    k = Y.shape[1]
    m = 50
    print('W1', k, d)
    
    W1 = np.random.randn(m, d) / np.sqrt(d)
    W2 = np.random.randn(k, m) / np.sqrt(m)
    # Initialize biases
    b1 = np.zeros((m, 1))
    b2 = np.zeros((k, 1))
    return W1, b1, W2, b2

def LoadBatch(filename):
    data = unpickle('/home/mech/sanchis/sanchis/PHD_COURSES/Deep_learning/Assign_1/DirName/data/cifar-10-batches-py/' + filename)
    # print(data.keys())
    X = np.array(data[b'data']).astype(np.float64)
    y = np.array(data[b'labels']).astype(np.float64).flatten()
    encoder = OneHotEncoder(sparse=False)
    Y = encoder.fit_transform(np.array(y).reshape(-1, 1))

    return X, Y, y

    return X, Y, y
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def preprocess_data(X_trn, X_tst, X_val):

  mu    = np.mean(X_trn, axis=0).reshape(1,-1)
  sigma = np.std(X_trn, axis=0).reshape(1,-1)

  X_trn = (X_trn - mu)/sigma
  X_tst = (X_tst - mu)/sigma
  X_val = (X_val - mu)/sigma

  return X_trn, X_tst, X_val

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
    
######### CALCULATIONS ##########

def ComputeCost(X, Y, W1, b1, W2, b2, lamda, return_loss=False):
    """
    Compute the cost function: c = loss + regularisation 

    Args: 
        X       : [d,n] input 
        Y       : [K,n] One-Hot Ground Truth 
        W       : [K,d] Weight 
        b       : [d,1] bias 
        lamb    : (float) a regularisation term for the weight
    
    Return:
        J       : (float) A scalar of the cost function 
    """
    p,h = EvaluateClassifier(X, W1, b1, W2, b2)  # Assuming EvaluateClassifier is defined elsewhere

    p_clipped = np.clip(p, 1e-16, 1 - 1e-16)
    p = np.clip(p,1e-16,1-1e-16)
    loss_cross = -np.mean(np.sum(Y * np.log(p_clipped.T), axis=1))

    reg = lamda * (np.sum(W1**2) + np.sum(W2**2))

    Total = loss_cross + reg
    # print(f"total :{Total},reg: {reg}")
    if return_loss:
        return Total, loss_cross
    else: 
        return Total
    
def ComputeCost_train(X, Y, W1, b1, W2, b2, lamda,p ,return_loss=False):
    """
    Compute the cost function: c = loss + regularisation 

    Args: 
        X       : [d,n] input 
        Y       : [K,n] One-Hot Ground Truth 
        W       : [K,d] Weight 
        b       : [d,1] bias 
        lamb    : (float) a regularisation term for the weight
    
    Return:
        J       : (float) A scalar of the cost function 
    """
    p_clipped = np.clip(p, 1e-16, 1 - 1e-16)
    p = np.clip(p,1e-16,1-1e-16)
    loss_cross = -np.mean(np.sum(Y * np.log(p_clipped.T), axis=1))

    reg = lamda * (np.sum(W1**2) + np.sum(W2**2))

    Total = loss_cross + reg
    # print(f"total :{Total},reg: {reg}")
    if return_loss:
        return Total, loss_cross
    else: 
        return Total

def compute_acc(X, Y, W1, b1, W2, b2,h,P):
    """
    Compute the accuracy of the classification 

    Args:
        X   : [d,n] input 
        Y   : [1,n] OR [d,n] Ground Truth 
        
    Returns: 
        acc : (float) a scalar value containing accuracy 
    """
    #P, h = EvaluateClassifier(X, W1, b1, W2, b2)  # Assuming EvaluateClassifier is defined elsewhere
    
    lenY = Y.shape[0]
    
    # Generate Output with [K,n]
    # Compute the maximum probability 
    P = np.clip(P, 1e-16, 1 - 1e-16)
    P = np.argmax(P, axis=0)
    # Compute how many true-positive samples
    if Y.shape[0] != 1:
        Y = np.argmax(Y, axis=1)

    true_pos = np.sum(P == Y)
    
    # Percentage on total 
    acc = true_pos / lenY
    
    del P, Y
    
    return acc


def softmax(s):
  return np.exp(s)/np.sum(np.exp(s), axis=0)

def relu(x):
	return np.maximum(0, x)

def EvaluateClassifier(X, W1, b1,W2, b2):
    # X: Input data matrix of size (d, N)
    # W: Weights matrix of size (m, d)
    # b: Bias vector of size (m, 1)

    # Compute the intermediate values
    s1 = W1@ X.T + b1  # Equation (1)
    h = relu(s1)  # Equation (2)
    s1 = W2@ h + b2  # Equation (3)

    # Compute softmax probabilities
    s1 = softmax(s1)  # Equation (4)

    return s1,h

######### GRADIENTS ############3
def ComputeGradients(X, Y, P, W1, b1, W2, b2, lamda,h,h1):
    n = X.shape[0]
    k = Y.shape[1]
    m = W1.shape[0]
    G = -(Y.T - P) # Gradient of the cross-entropy loss with respect to s (before softmax)
    # Compute the gradient of the weights (W1 and W2)
    grad_W2 = (G @ relu(h).T)  # Gradient of the loss with respect to W2
    grad_b2 = np.sum(G,axis=1,keepdims=True)


    # Compute the gradient of the biases (b1 and b2)
    G = W2.T@G
    G[h<=0]=0
    grad_W1 = G@X

    grad_b1 = np.sum(G,axis=1,keepdims=True) # Gradient of the loss with respect to b1)
    
    return grad_W1, grad_b1, grad_W2, grad_b2

def ComputeGradsNumSlow(X, Y,W1, b1, W2, b2,lamda, h):
	""" 
	Converted from matlab code 
	
	"""

	grad_W1 = np.zeros_like(W1);
	grad_b1 = np.zeros_like(b1);
	grad_W2 = np.zeros_like(W2);
	grad_b2 = np.zeros_like(b2);

	for i in range(W2.shape[0]):
		for j in range(W2.shape[1]):
			W_try = np.array(W2)
			W_try[i,j] -= h
			c1 = ComputeCost(X,Y,W1,b1,W_try,b2,0)
			W_try = np.array(W2)
			W_try[i,j] += h
			c2 = ComputeCost(X,Y,W1,b1,W_try,b2,0)
			grad_W2[i,j] = (c2-c1) / (2*h)
	print("INFO: Compute W2 Grad")

	
	for i in range(len(b2)):
		b_try = np.array(b2)
		b_try[i] -= h
		c1 = ComputeCost(X,Y,W1,b1,W2,b_try,0)
		b_try = np.array(b2)
		b_try[i] += h
		c2 = ComputeCost(X,Y,W1,b1,W2,b_try,0)
		grad_b2[i] = (c2-c1) / (2*h)
	print("INFO: Compute b2 Grad")


	
	for i in range(W1.shape[0]):
		for j in range(W1.shape[1]):
			W_try = np.array(W1)
			W_try[i,j] -= h
			c1 = ComputeCost(X,Y,W_try,b1,W2,b2,0)
			W_try = np.array(W1)
			W_try[i,j] += h
			c2 = ComputeCost(X,Y,W_try,b1,W2,b2,0)
			grad_W1[i,j] = (c2-c1) / (2*h)
	print("INFO: Compute W1 Grad")

		
	for i in range(len(b1)):
		b_try = np.array(b1)
		b_try[i] -= h
		c1 = ComputeCost(X,Y,W1,b_try,W2,b2,0)
		b_try = np.array(b1)
		b_try[i] += h
		c2 = ComputeCost(X,Y,W1,b_try,W2,b2,0)
		grad_b1[i] = (c2-c1) / (2*h)
	print("INFO: Compute b1 Grad")
	
	return grad_W1, grad_b1, grad_W2, grad_b2

def ComputeGradsNum(X, Y, W1, b1, W2, b2,lamda, h):
	""" 
	Converted from matlab code 
	
	"""

	grad_W1 = np.zeros_like(W1);
	grad_b1 = np.zeros_like(b1);
	grad_W2 = np.zeros_like(W2);
	grad_b2 = np.zeros_like(b2);

	c1 = ComputeCost(X,Y,W1,b1,W2,b2,0)
			
	for i in range(W2.shape[0]):
		for j in range(W2.shape[1]):
			W_try = np.array(W2)
			W_try[i,j] += h
			c2 = ComputeCost(X,Y,W1,b1,W_try,b2,0)
			grad_W2[i,j] = (c2-c1) /h
	print("INFO: Compute W2 Grad")

	
	for i in range(len(b2)):
		b_try = np.array(b2)
		b_try[i] += h
		c2 = ComputeCost(X,Y,W1,b1,W2,b_try,0)
		grad_b2[i] = (c2-c1) / (h)
	print("INFO: Compute b2 Grad")


	
	for i in range(W1.shape[0]):
		for j in range(W1.shape[1]):
			W_try = np.array(W1)
			W_try[i,j] += h
			c2 = ComputeCost(X,Y,W_try,b1,W2,b2,0)
			grad_W1[i,j] = (c2-c1) / (h)
	print("INFO: Compute W1 Grad")

		
	for i in range(len(b1)):
		b_try = np.array(b1)
		b_try[i] += h
		c2 = ComputeCost(X,Y,W1,b_try,W2,b2,0)
		grad_b1[i] = (c2-c1) / (h)
	print("INFO: Compute b1 Grad")
	

	
	return [grad_W1, grad_b1, grad_W2, grad_b2]


############ ERRORS ##############3
def Prop_Error(ga,gn,eps):
	"""
	Compute the propagation Error with uncertainty
	"""

	eps_m = np.ones_like(ga).astype(np.float64) * eps
	n,m = ga.shape
	summ  = np.abs(ga)+np.abs(gn)

	return np.abs(ga-gn)/np.maximum(eps_m,summ)

def propagate(x, y, eta_, batch_size, P_small, W1, b1, W2, b2, lamda, h, h1):
    """Back Prop for Update the W&B"""
    grad_W1, grad_b1, grad_W2, grad_b2 = ComputeGradients(x, y, P_small, W1, b1, W2, b2, lamda, h, h1)

    W1 -= eta_ * ((1 / batch_size) * (grad_W1) + 2 * lamda * W1)
    b1 -= eta_ * ((1 / batch_size) * (grad_b1))

    W2 -= eta_ * ((1 / batch_size) * (grad_W2) + 2 * lamda * W2)
    b2 -= eta_ * ((1 / batch_size) * (grad_b2))

    return W1, b1, W2, b2

########### TRAINING ##############
def train(X, Y, X_val, Y_val,  W1, b1, W2, b2, n_epochs, n_batch,h1, lamda, lr_sch, fix_eta=1e-2):
    print(f'Start Training, Batch Size = {n_batch}')
    lenX = X.T.shape[-1]
    lenX_val = X_val.T.shape[-1]
    batch_size = n_batch
    train_batch_range = np.arange(0, lenX // n_batch)
    hist = {}
    hist['train_cost'] = []
    hist['val_cost'] = []
    hist['train_loss'] = []
    hist['val_loss'] = []
    hist["train_acc"] = []
    hist["val_acc"] = []
    st = time.time()
    for epoch in tqdm(range(n_epochs)): 
        epst = time.time()
        # Shuffle the batch indices 
        indices = np.random.permutation(lenX)
        X_ = X[indices, :]
        Y_ = Y[indices,:]
        for b in train_batch_range:
            eta_ = (lr_sch.eta if lr_sch is not None else fix_eta)

            X_batch = X_[b * batch_size:(b + 1) * batch_size, :]
            Y_batch = Y_[b * batch_size:(b + 1) * batch_size,:]
            
            P_small, h = EvaluateClassifier(X_batch, W1, b1, W2, b2)  # Assuming EvaluateClassifier is defined elsewhere
            W1, b1,W2,b2 = propagate(X_batch, Y_batch, eta_, batch_size, P_small, W1, b1, W2, b2, lamda, h,h1)  # Assuming propagate is defined elsewhere

            # Compute the cost func and loss func
            jc, l_train = ComputeCost_train(X_batch, Y_batch, W1, b1, W2, b2, lamda,P_small,return_loss=True)# Assuming ComputeCost is defined elsewhere
            hist['train_cost'].append(jc)
            hist['train_loss'].append(l_train)
            
            P_val, h_val = EvaluateClassifier(X_val, W1, b1, W2, b2)
            jc_val, l_val = ComputeCost_train(X_val, Y_val, W1, b1, W2, b2, lamda,P_val,return_loss=True)
            hist['val_cost'].append(jc_val)
            hist['val_loss'].append(l_val)

            # Compute the accuracy 
            train_acc = compute_acc(X_batch, Y_batch, W1, b1, W2, b2,h,P_small)  # Assuming compute_acc is defined elsewhere
            val_acc = compute_acc(X_val, Y_val, W1, b1, W2, b2,h,P_val)
            hist["train_acc"].append(train_acc)
            hist["val_acc"].append(val_acc)

            if lr_sch is not None:
                lr_sch.update_lr()

        epet = time.time()
        epct = epet - epst

        if lr_sch is not None: 
            print(f"\n Epoch ({epoch+1}/{n_epochs}), At Step =({lr_sch.t}/{n_epochs*lenX//batch_size}), Cost Time = {epct:.2f}s\n"+\
                f" Train Cost ={hist['train_cost'][-1]:.3f}, Val Cost ={hist['val_cost'][-1]:.3f}\n"+\
                f" Train Loss ={hist['train_loss'][-1]:.3f}, Val Loss ={hist['val_loss'][-1]:.3f}\n"+\
                f" Train Acc ={hist['train_acc'][-1]:.3f}, Val Acc ={hist['val_acc'][-1]:.3f}\n"+\
                f" The LR = {lr_sch.eta:.4e}")
        else:
            print(f"\n Epoch ({epoch+1}/{n_epochs}), Cost Time = {epct:.2f}s\n"+\
                f" Train Cost ={hist['train_cost'][-1]:.3f}, Val Cost ={hist['val_cost'][-1]:.3f}\n"+\
                f" Train Loss ={hist['train_loss'][-1]:.3f}, Val Loss ={hist['val_loss'][-1]:.3f}\n"+\
                f" Train Acc ={hist['train_acc'][-1]:.3f}, Val Acc ={hist['val_acc'][-1]:.3f}\n"+\
                f" The LR = {fix_eta:.4e}")

    et =  time.time()
    cost_time = et - st 
    print(f"INFO: Training End, Cost Time = {cost_time:.2f}")
    return hist

#%%
print('-------------- LOADING DATA & PRE-PROCESS-----------')
data_batch1 = "data_batch_1"
data_batch2 = "data_batch_2"
test_batch  = "test_batch"
path = "/home/mech/sanchis/sanchis/PHD_COURSES/Deep_learning/Assign_2/results/"

X_trn, Y_trn, y_trn = LoadBatch(data_batch1)
X_val, Y_val, y_val = LoadBatch(data_batch2)
X_tst, Y_tst, y_tst = LoadBatch(test_batch)

X_trn, X_tst, X_val = preprocess_data(X_trn, X_tst, X_val)

montage(X_trn)
#%%


print('---------- INITIALIZED -------------------')
lamda = 0
batch_size  = 1
trunc 		= 20
X_small = X_trn[:batch_size,:trunc]
Y_small = Y_trn[:batch_size,:]
h1 			= 1e-5 # Given in assignmen
W1, b1, W2, b2 = initialize_parameters(X_small, Y_small)

P_small,h = EvaluateClassifier(X_small,W1,b1,W2,b2)

print('INPUTS', Y_small.shape)
anl_grad_W1, anl_grad_b1,anl_grad_W2, anl_grad_b2 = ComputeGradients(X_small, Y_small, P_small, W1, b1,W2, b2, lamda,h,h1)

grad_error = {}
num_grad_W1, num_grad_b1,num_grad_W2, num_grad_b2 =  ComputeGradsNumSlow(X_small, Y_small, W1, b1,W2,b2, lamda, h1)

print("Central Method")
ew = Prop_Error(anl_grad_W1,num_grad_W1,h1)
eb = Prop_Error(anl_grad_b1,num_grad_b1,h1)
print(f"Comparison: Prop Error for W1:{ew.mean():.3e}")
print(f"Comparison: Prop Error for B1:{eb.mean():.3e}")
grad_error["central_w1"] = ew.mean().reshape(-1,)
grad_error["central_b1"] = eb.mean().reshape(-1,)

print("Central Method")
ew = Prop_Error(anl_grad_W2,num_grad_W2,h1)
eb = Prop_Error(anl_grad_b2,num_grad_b2,h1)
print(f"Comparison: Prop Error for W2:{ew.mean():.3e}")
print(f"Comparison: Prop Error for B2:{eb.mean():.3e}")
grad_error["central_w2"] = ew.mean().reshape(-1,)
grad_error["central_b2"] = eb.mean().reshape(-1,)

nums_grad_W1, nums_grad_b1,nums_grad_W2, nums_grad_b2 =  ComputeGradsNum(X_small, Y_small, W1, b1,W2,b2, lamda, h1)

print("Implicit Method")
ew = Prop_Error(anl_grad_W1,nums_grad_W1,h1)
eb = Prop_Error(anl_grad_b1,nums_grad_b1,h1)
print(f"Comparison: Prop Error for W1:{ew.mean():.3e}")
print(f"Comparison: Prop Error for B1:{eb.mean():.3e}")
grad_error["central_w1"] = ew.mean().reshape(-1,)
grad_error["central_b1"] = eb.mean().reshape(-1,)

print("Central Method")
ew = Prop_Error(anl_grad_W2,nums_grad_W2,h1)
eb = Prop_Error(anl_grad_b2,nums_grad_b2,h1)
print(f"Comparison: Prop Error for W2:{ew.mean():.3e}")
print(f"Comparison: Prop Error for B2:{eb.mean():.3e}")
grad_error["central_w2"] = ew.mean().reshape(-1,)
grad_error["central_b2"] = eb.mean().reshape(-1,)
#%%
######### SMALL TRAINING ###########
lamda = 0
trunc = 100
n_epochs = 200
batch = 10
lamda = 0
X_small = X_trn[:,:trunc]
Y_small = Y_trn[:,:]
X_val = X_val[:,:trunc]
Y_val = Y_val[:,:]
h1 			= 1e-5 # Given in assignmen
W1, b1, W2, b2 = initialize_parameters(X_small, Y_small)
hist = train(X_small,Y_small,X_val,Y_val,W1,b1,W2,b2,n_epochs,batch,h1,lamda,lr_sch=None,fix_eta = 1e-3)
#%%
########### PLOTTING ##########
fig, axs = plot_hist(hist,0,1)
for ax in axs:
	ax.set_xlabel('Update Steps',font_dict)
fig.savefig('Figs/validate_mini_batch.jpg',**fig_dict)

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
casename=['0.001', '0.01', '1']
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