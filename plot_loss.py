import numpy as np
import matplotlib.pyplot as plt

def plot_loss():
    '''Plots Huber loss function and squared error loss function.
    Currently hard coded to t=0 and delta=5
    
    '''
    delta = 5
    xlim = 3*delta
    step = xlim/50
    
    x1 = np.arange(-xlim, xlim, step)
    x2 = np.arange(-(delta+step), (delta+step), step)
    x3 = np.arange(delta, xlim, step)
    x4 = np.arange(-xlim, -delta, step)
    
    y1 = 0.5*x1**2
    y2 = 0.5*x2**2
    y3 = delta*(np.abs(x3) - 0.5*delta)
    y4 = delta*(np.abs(x4) - 0.5*delta)
    
    fig, ax = plt.subplots()
    ax.plot(x1,y1,'r-', label = 'Squared error loss')
    ax.plot(x2,y2,'b-', label = 'Huber loss')
    ax.plot(x3,y3,'b-')
    ax.plot(x4,y4,'b-')
    fig.suptitle('t = 0, delta = 5')
    plt.xlabel('y')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    # plt.savefig('huber-squared-loss.png')


if __name__ == '__main__':
    # plot_loss()
    
    
    