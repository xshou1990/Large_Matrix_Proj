import numpy as np
from timeit import default_timer as timer


"""adagrad full for least squares

Parameters:
Z is dataset in numpy
y is labels in numpy
Tmax is max iteration
delta is the tolerance 
eta is stepsize
batchsize the batch size for epoch

""" 
def adagradLS(Z, y,Tmax,delta,eta,batchsize):
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
    G0 = np.zeros((d+1,d+1))
    
    # number of epochs
    n_epoch = int(n/batchsize)
    
    loss_hist = np.array([])
    time = np.array([])
    
    # main iteration
    while t < Tmax :
        j = np.random.randint(1,n_epoch+1)
        loss = cost(Z,y,x1)
        subg = subgrad2(Z[((j-1)*batchsize):(j*batchsize),:],y[((j-1)*batchsize):(j*batchsize)],x1)
        # adagrad full update
        Gt = G0 + subg*subg.T + delta*np.identity(d+1)
        u, s, vh = np.linalg.svd(Gt, full_matrices=True)
        St = np.matmul(np.matmul(u,np.diag(np.sqrt(s))),vh)
        Ht = St + delta*np.identity(d+1)
        x = x1 + np.linalg.solve(Ht,-eta*subg)
        
        x1 = x
        G0=Gt
        t = t + 1
        loss_hist = np.append(loss_hist,loss)
        time_count = timer() - start
        time = np.append(time,time_count)
        
    return loss_hist,time

def subgrad2(Z,y,x):
    n = Z.shape[0]
    d= Z.shape[1]
    Znew = np.append(Z, np.ones((n,1)),axis=1)
    diff = y - np.matmul(Znew,x)
    return -2*np.matmul(Znew.T,diff)

def cost(Z,y,x):
    n = Z.shape[0]
    d = Z.shape[1]
    bias = np.ones((n,1))
    Znew = np.append(Z,bias,axis=1)
    loss = np.linalg.norm(y-np.matmul(Znew,x),ord=2)
    return loss