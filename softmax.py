"""Softmax."""
import numpy as np

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    x = np.array(x)
    if x.ndim == 1 :
        return np.exp(x)/np.exp(x).sum()
    elif x.ndim == 2:
        for i in range(0,x.shape[1]):  #to iterate between 10 to 20
            x_col = x[:,i] 
            out_col = np.exp(x_col)/np.exp(x_col).sum()
            if i==0:
                out = out_col
            else:
                out = np.vstack([out, out_col])
        print ( np.transpose(out) )
        return np.transpose(out)
    else:
        print ( "Error, input is not 1D or 2D dimensions")


# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])
plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()


# Array practice
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])
c = np.vstack((a,b))
array([[1, 2, 3],
       [2, 3, 4]])
    
    
arrays = [np.random.randn(3, 4) for _ in range(10)]



change = [1, 'pennies', 2, 'dimes', 3, 'quarters']
for i in change:
    print("I got %r" % i	)
