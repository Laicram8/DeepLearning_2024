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
# For visualisation 
font_dict = {'size':20,'weight':'bold'}
fig_dict = {'bbox_inches':'tight','dpi':300}
class colorplate:
    red = "#FF2400"   # Red
    blue = "#00FFFF"  # Cyan
    yellow = "#FFD700" # Gold
    cyan = "#00BFFF"  # Deep Sky Blue
    black = "#000000" # Black
    gray = "#808080"  # Gray

plt.rc("font",family = "serif")
plt.rc("font",size = 22)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 18)
plt.rc("ytick",labelsize = 18)

#############3 DATA ################
def montage_seq(W,c,epoch):
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
    plt.savefig(path + f'train/weights_f{c}_{epoch}_batch_size.png')
    plt.show()
    
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
    
    return W1, b1, W2, b2  # Return the updated weights


########### TRAINING ##############
def train(X, Y, X_val, Y_val, W1, b1, W2, b2, n_epochs, n_batch, h1, lamda, lr_sch, fix_eta=1e-3):
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

    W1_hist = []
    W2_hist = []
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
            
            P_small, h_small = EvaluateClassifier(X_batch, W1, b1, W2, b2)  # Assuming EvaluateClassifier is defined elsewhere
            W1, b1,W2,b2 = propagate(X_batch, Y_batch, eta_, batch_size, P_small, W1, b1, W2, b2, lamda, h_small,h1)
            """
            grad_W1, grad_b1, grad_W2, grad_b2 = ComputeGradients(X_batch, Y_batch, P_small, W1, b1, W2, b2, lamda, h_small, h1)

            W1 -= eta_ * ((1 / batch_size) * (grad_W1) + 2 * lamda * W1)
            b1 -= eta_ * ((1 / batch_size) * (grad_b1))

            W2 -= eta_ * ((1 / batch_size) * (grad_W2) + 2 * lamda * W2)
            b2 -= eta_ * ((1 / batch_size) * (grad_b2))
            """
            if lr_sch is not None:
                lr_sch.update_lr()
                
            if 0 % batch_size == 0:
                #W1, b1,W2,b2 = propagate(X_batch, Y_batch, eta_, batch_size, P_small, W1, b1, W2, b2, lamda, h,h1)  # Assuming propagate is defined elsewhere
                # Compute the cost func and loss func
                P, h = EvaluateClassifier(X, W1, b1, W2, b2)
                jc, l_train = ComputeCost_train(X, Y, W1, b1, W2, b2, lamda,P,return_loss=True)# Assuming ComputeCost is defined elsewhere
                hist['train_cost'].append(jc)
                hist['train_loss'].append(l_train)
                
                P_val, h_val = EvaluateClassifier(X_val, W1, b1, W2, b2)
                jc_val, l_val = ComputeCost_train(X_val, Y_val, W1, b1, W2, b2, lamda,P_val,return_loss=True)
                hist['val_cost'].append(jc_val)
                hist['val_loss'].append(l_val)
    
                # Compute the accuracy 
                train_acc = compute_acc(X, Y, W1, b1, W2, b2,h,P)  # Assuming compute_acc is defined elsewhere
                val_acc = compute_acc(X_val, Y_val, W1, b1, W2, b2,h_val,P_val)
                hist["train_acc"].append(train_acc)
                hist["val_acc"].append(val_acc)
        
        # Store weights for plotting
        W1_hist.append(W1.copy())
        W2_hist.append(W2.copy())
        # Plot weights after every epoch
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
    return hist, W1,b1,W2,b2

class lr_scheduler:
	def __init__(self,eta_max,eta_min,n_s):
		"""
	cyclinal learning rate during training
	
	Args: 
		t: (int) Current Number of iteration 

		eta_min: Lower bound of learning rate 

		eta_max: Upper bound of learning rate 

		n_s		: How many epoch per cycle 
		"""

		self.eta_min = eta_min
		self.eta_max = eta_max
		self.n_s 	 = n_s
		self.eta 	 = eta_min
		self.hist    = []
		self.t 		 = 0
		print(f"INFO: LR scheduler:"+\
			f"\n eta_min={self.eta_min:.2e}, eta_max={self.eta_max:.2e}, n_s={self.n_s}"	)
	def update_lr(self):
		"""
		Update the LR
		
		"""
		# cycle = np.floor(1+self.t/(2*self.n_s))
		cycle = (self.t//(2*self.n_s))
		# x = abs(self.t / self.step_size - 2 * cycle + 1)
		
		if (2 * cycle * self.n_s <= self.t) and (self.t <= (2 * cycle + 1) * self.n_s):
			
			self.eta=self.eta_min+(self.t-2*cycle*self.n_s)/\
					self.n_s*(self.eta_max-self.eta_min)
		
		elif ((2 * cycle +1) * self.n_s <= self.t) and (self.t <= 2*( cycle + 1) * self.n_s) :
			
			self.eta=self.eta_max-(self.t-(2*cycle+1)*self.n_s)/\
					self.n_s*(self.eta_max-self.eta_min)
		
		self.hist.append(self.eta)
		self.t +=1

########## PLOTTIN ##############
import seaborn as sns

def plot_weights(W1_hist, W2_hist, epoch):
    """Plot weights after every epoch as heatmaps"""
    fig, axs = plt.subplots(2, 1)
    fig.suptitle(f'Weights After Epoch {epoch + 1}')
    
    # Convert W1_hist[-1] and W2_hist[-1] to numpy arrays if they are not already
    W1 = np.array(W1_hist)
    W2 = np.array(W2_hist)
    # Check if W1 and W2 are 2D arrays
    if W1.ndim != 2 or W2.ndim != 2:
        raise ValueError("W1 and W2 should be 2D arrays")
    
    # Plot W1 as heatmap
    sns.heatmap(W1, ax=axs[0], cmap='coolwarm', center=0)
    
    # Plot W2 as heatmap
    sns.heatmap(W2, ax=axs[1], cmap='coolwarm', center=0)
    plt.show()

def plot_hist(hist,n_start,n_interval):
	fig , axs  = plt.subplots(1,3,figsize=(24,6))
	
	n_range = np.arange(len(hist['train_cost']))

	fig, axs[0] = plot_loss(n_range[n_start:-1:], hist['train_cost'][n_start:-1:],fig,axs[0],color = colorplate.red,ls = '-')
	fig, axs[0] = plot_loss(n_range[n_start:-1:], hist['val_cost'][n_start:-1:],fig,axs[0],color =colorplate.blue, ls = '-')
	axs[0].set_ylabel('Cost',font_dict)

	fig, axs[1] = plot_loss(n_range[n_start:-1:],hist['train_loss'][n_start:-1:],fig,axs[1],color = colorplate.red,ls = '-')
	fig, axs[1] = plot_loss(n_range[n_start:-1:],hist['val_loss'][n_start:-1:],fig,axs[1],color =colorplate.blue, ls = '-')
	axs[1].set_ylabel('Loss',font_dict)
	
	fig, axs[2] = plot_loss(n_range[n_start:-1:],hist['train_acc'][n_start:-1:],fig,axs[2],color =colorplate.red, ls = '-')
	fig, axs[2] = plot_loss(n_range[n_start:-1:],hist['val_acc'][n_start:-1:],fig,axs[2],color =colorplate.blue, ls = '-')
	axs[2].set_ylabel('Accuracy',font_dict)
	
	for ax in axs:
		ax.legend(['Train',"Validation"],prop={'size':20})
	return fig, axs 

def name_case(n_s,eta_min,eta_max,n_batch,n_epochs,lamda):
	case_name = f"W&B_{n_batch}BS_{n_epochs}Epoch_{n_s}NS_{lamda:.3e}Lambda_{eta_min:.3e}MINeta_{eta_max:.3e}MAXeta"
	return case_name

def save_as_mat(W1,b1,W2,b2,hist,acc,name):
	""" Used to transfer a python model to matlab """
	import scipy.io as sio
	sio.savemat(name + '.mat',
			{
				"W1":W1,
				"W2":W2,
				"b1":b1,
				"b2":b2,
				'train_loss':np.array(hist['train_loss']),
				'train_cost':np.array(hist['train_cost']),
				'train_acc':np.array(hist['train_acc']),
				'val_loss':np.array(hist['val_loss']),
				'val_cost':np.array(hist['val_loss']),
				'val_acc':np.array(hist['val_acc']),
				"test_acc":acc
				})

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
trunc = 200
n_epochs = 200
batch = 10
lamda = 0
ex = 100
lr_dict = {"n_s":500,"eta_min":1e-5,"eta_max":1e-1} 
X_small = X_trn[:ex,:trunc]
Y_small = Y_trn[:ex,:]
X_v = X_val[:ex,:trunc]
Y_v = Y_val[:ex,:]
h1 			= 1e-5 # Given in assignmen
W1, b1, W2, b2 = initialize_parameters(X_small, Y_small)
lr_sch = lr_scheduler(**lr_dict)
plot_weights(W1,W2,-1)
hist, W1,b1,W2,b2 = train(X_small,Y_small,X_v,Y_v,W1,b1,W2,b2,n_epochs,batch,h1,lamda,lr_sch=None,fix_eta = 1e-2)
#%%
########### PLOTTING & SAVING ##########
fig, axs = plot_hist(hist,0,1)
for ax in axs:
	ax.set_xlabel('Update Steps',font_dict)
plt.savefig(path + '/training.png', bbox_inches="tight")




#%%%
print('------------ ASSIGNMENT STAGE -------------')
lr_dict = {"n_s":500,"eta_min":1e-5,"eta_max":1e-1} 
etas = []
lr_sch = lr_scheduler(**lr_dict)
n_epoch = 10
for l in range(n_epoch):
    for b in range(100):
        lr_sch.update_lr()

fig, axs = plt.subplots(1,1,figsize=(6,4))
print(f"Maximum eta= {max(lr_sch.hist)}, Minimum eta = {min(lr_sch.hist)}")
axs.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
axs.plot(range(len(lr_sch.hist)),lr_sch.hist,'o',markersize=1.5,c=colorplate.red,lw=2)
axs.set_yticks([1e-5,1e-1])
axs.set_xlabel('Step',font_dict)
axs.set_ylabel(r'$\eta_t$',font_dict)
fig.savefig(path + 'Figs/LR_schedule_1.jpg',bbox_inches='tight',dpi=300)

lr_dict = {"n_s":800,"eta_min":1e-5,"eta_max":1e-1} 
etas = []
lr_sch = lr_scheduler(**lr_dict)
n_epoch = 48
for l in range(n_epoch):
    for b in range(100):
        lr_sch.update_lr()
        
fig, axs = plt.subplots(1,1,figsize=(6,4))
print(f"Maximum eta= {max(lr_sch.hist)}, Minimum eta = {min(lr_sch.hist)}")
axs.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1e'))
	
axs.plot(range(len(lr_sch.hist)),lr_sch.hist,'o',markersize=1.5,c=colorplate.cyan,lw=2)
axs.set_yticks([1e-5,1e-1])
axs.set_xlabel('Update Step',font_dict)
axs.set_ylabel(r'$\eta_t$',font_dict)
fig.savefig(path + 'Figs/LR_schedule_2_log.jpg',bbox_inches='tight',dpi=300)



#%%
########### ex 3 ###########
print('------------ eXERCISE 3 -------------')
lr_dict = {"n_s":500,"eta_min":1e-5,"eta_max":1e-1} 
train_dict = {'n_batch':100,'n_epochs':10}
batch = 100
n_epochs = 10
lamda  = 0.01
h1 = 1e-5 # Given in assignment
filename = name_case(**lr_dict, **train_dict, lamda=lamda)
lr_sch = lr_scheduler(**lr_dict)
W1, b1, W2, b2 = initialize_parameters(X_trn, Y_trn)
hist, W1, b1, W2, b2 = train(X_trn, Y_trn, X_val, Y_val, W1, b1, W2, b2, n_epochs, batch, h1, lamda, lr_sch, fix_eta=1e-3)
P, h = EvaluateClassifier(X_tst, W1, b1, W2, b2)
acc = compute_acc(X_tst, Y_tst, W1, b1, W2, b2, h, P)
print(f"Acc ={acc*100}")
print("#"*30)
save_as_mat(W1,b1,W2,b2, hist, acc, "weights/" + filename)
print(f"W&B Saved!")
fig, axs = plot_hist(hist, 10, batch)
for ax in axs:
    ax.set_xlabel('Update Step')
fig.savefig(path + f'Figs/Loss_{filename}.jpg', **fig_dict)
#%%%
print('------------ eXERCISE 4 -------------')
lr_dict = {"n_s":800,"eta_min":1e-5,"eta_max":1e-1} 
n_cycle = 3 
batch = 100
n_epochs = int(n_cycle * lr_dict['n_s'] * 2 / batch)
train_dict = {'n_batch':batch,'n_epochs':n_epochs}
lamda  = 0.01
h1 = 1e-5 # Given in assignment
filename = name_case(**lr_dict, **train_dict, lamda=lamda)
lr_sch = lr_scheduler(**lr_dict)
W1, b1, W2, b2 = initialize_parameters(X_trn, Y_trn)
hist, W1, b1, W2, b2 = train(X_trn, Y_trn, X_val, Y_val, W1, b1, W2, b2, n_epochs, batch, h1, lamda, lr_sch, fix_eta=1e-3)
P, h = EvaluateClassifier(X_tst, W1, b1, W2, b2)
acc = compute_acc(X_tst, Y_tst, W1, b1, W2, b2, h, P)
print(f"Acc ={acc*100}")
print("#"*30)
save_as_mat(W1,b1,W2,b2, hist, acc, "weights/" + filename)
print(f"W&B Saved!")
fig, axs = plot_hist(hist, 10, batch)
for ax in axs:
    ax.set_xlabel('Update Step')
fig.savefig(path + f'Figs/Loss_{filename}.jpg', **fig_dict)
#%%
print('------------ Best STAGE -------------')
X_trn, Y_trn, y_trn = LoadBatch(data_batch1)
X_val, Y_val, y_val = LoadBatch(data_batch2)
X_tst, Y_tst, y_tst = LoadBatch(test_batch)

X_trn, X_tst, X_val = preprocess_data(X_trn, X_tst, X_val)

#%%