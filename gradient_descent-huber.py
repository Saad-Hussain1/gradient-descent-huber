import numpy as np

def gradient_descent(X, y, lr, num_iter, delta):
    '''Performs gradient descent according to the Huber loss function
    Input:  X is the N x d design matrix
            y is the target vector of shape (N,)
            lr is the learning rate
            num_iter is the no. of iterations to run gradient descent
            delta is the hyperparameter for the Huber loss
    Output is w = weights as dx1 vector and b = bias as scalar
    '''
    y = y.reshape(y.size, 1) #turn y into column vector for computations
    N = X.shape[0]
    d = X.shape[1]
    w = np.zeros((d,1)) #init weights
    b = 0 #init bias
    dw = np.zeros((N,1)) #init gradient of loss wrt w
    db = np.zeros((N,1)) #init gradient of loss wrt b
    for i in range(num_iter):
        h = X @ w + b
        
        cond1 = np.where(h - y < -delta)
        cond2 = np.where((-delta <= h - y) * (h - y <= delta))
        cond3 = np.where(h - y > delta)
        
        dw[cond1] = -delta * X[cond1]
        dw[cond2] = (h-y)[cond2] * X[cond2]
        dw[cond3] = delta * X[cond3]
        w -= lr * 1/N * np.sum(dw, axis=0)
        w = w.reshape(w.size,1)
        
        db[cond1] = -delta
        db[cond2] = (h-y)[cond2]
        db[cond3] = delta
        b -= lr * 1/N * np.sum(db)
    return w, b

    
    
    