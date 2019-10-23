import numpy as np
import time

def calculate_hypothesis(X, theta, i):
    """
        :param X            : 2D array of our dataset
        :param theta        : 1D array of the trainable parameters
        :param i            : scalar, index of current training sample's row
    """
    
    hypothesis = 0.0
    #########################################
    # Write your code here
    # You must calculate the hypothesis for the i-th sample of X, given X, theta and i.
    
    #########################################
    #hypothesis = theta
    # print(theta[0],theta[1])
    # print(X[0])
    #hypothesis = np.multiply(theta[0],X[i, 0]) + np.multiply(theta[1],X[i, 1])
    # hypothesis = X[i, 0] * theta[0] + X[i, 1] * theta[1]
    #hypothesis = theta[0]*X[0] + theta[1]*X[1]
    print(hypothesis)
    #time.sleep(1)
    return hypothesis
