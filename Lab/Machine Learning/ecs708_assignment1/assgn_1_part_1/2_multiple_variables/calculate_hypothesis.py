import numpy as np

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    
    #########################################
    # Write your code here
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    
    ########################################/
    # print(X[i].shape)#,theta)
    # hypothesis = X[i,0] * theta[0] + X[i,1] * theta[1] + X[i,2] * theta[2]
    hypothesis = np.dot(X[i],theta)
    return hypothesis
