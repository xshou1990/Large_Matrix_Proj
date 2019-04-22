import numpy as np
from timeit import default_timer as timer
"""
Created on Apr. 21, 2019

@author: Xiao
"""

"""adagrad diag for hinge loss with L1 regularization

Parameters:
Z is dataset in numpy
y is labels in numpy
lam is coefficient for L1
Tmax is max iteration
delta is the tolerance 
eta is stepsize
batchsize the batch size for epoch

""" 

def adagradregL1diag(Z, y, lam,Tmax,delta,eta,batchsize):
    start = timer()
    # shuffle data
    Zy = np.append(Z,y,axis=1)
    np.array(np.random.shuffle(Zy))
    Z,y = Zy[:,:-1],np.expand_dims(Zy[:,-1],axis=1)

    # n is the number of sample points
    n = Z.shape[0]
    # d is the dimension of each sample point
    d = Z.shape[1]
    
    # intialization
    t=0
    x1 = np.zeros((d+1,1))
    G0 = np.zeros((d+1,1))
    
    # set number of epochs 
    n_epoch = int(n/batchsize)
    
    loss_hist = np.array([])
    time = np.array([])
    
    # main iteration
    while t < Tmax :
        j = np.random.randint(1,n_epoch+1)
        loss = reghingeloss(Z,y,x1,lam)
        subg = subgrad(Z[((j-1)*batchsize):(j*batchsize),:],y[((j-1)*batchsize):(j*batchsize)],x1,lam)
        #adagrad updates
        Gt = np.append(G0,subg,axis=1)
        St = np.diag(np.linalg.norm(Gt, axis=1))
        Ht = St + delta*np.identity(d+1)
        # L1 shrinkage
        update = x1- eta*subg/np.expand_dims(np.diag(Ht),axis=1)
        skg = abs(update)-lam*eta/np.expand_dims(np.diag(Ht),axis=1)
        skg[skg<0]=0
        x = np.sign(update)*skg
        x1 = x
        G0=Gt
        t = t + 1
        loss_hist = np.append(loss_hist,loss)
        
        time_count = timer() - start
        time = np.append(time,time_count)
        
    return loss_hist,time

# loss function 
def reghingeloss(Z,y,x,lam):
    n = Z.shape[0]
    d = Z.shape[1]
    bias = np.ones((n,1))
    Znew = np.append(Z,bias,axis=1)
    diff = np.ones((n,1))-  y*(np.matmul(Znew,x))
    loss = 1/n*diff[diff>0].sum() + lam*np.linalg.norm(x,ord=1) 
    return loss

# subgrad of loss
def subgrad(Z,y,x,lam):
    n = Z.shape[0]
    d= Z.shape[1]
    L1grad = np.zeros((d+1,1))
    Znew = np.append(Z, np.ones((n,1)),axis=1)
    t = -y*Znew
    t[t<0] = 0
    fsubgrad = np.expand_dims(np.sum(t,axis=0),axis=1)
    L1grad[x<0] = -lam
    L1grad[x>=0] = lam
    subgradtot = 1/n*fsubgrad + L1grad
    return subgradtot
