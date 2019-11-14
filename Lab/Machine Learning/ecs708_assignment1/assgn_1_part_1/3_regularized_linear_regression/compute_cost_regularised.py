from calculate_hypothesis import *
import time
def compute_cost_regularised(X, y, theta, l):
    
    """
    :param X        : 2D array of our dataset
    :param y        : 1D array of the groundtruth labels of the dataset
    :param theta    : 1D array of the trainable parameters
    :param l        : scalar, regularization parameter
    """
    
    # initialize costs
    total_squared_error = 0.0
    total_regularised_error = 0.0
    
    # get number of training examples
    m = y.shape[0]
    
    for i in range(m):
        hypothesis = calculate_hypothesis(X, theta, i)
        output = y[i]
        squared_error = (hypothesis - output)**2
        total_squared_error += squared_error
    
    for i in range(1,len(theta)):
        total_regularised_error += theta[i]**2
    
    J = (total_squared_error + total_regularised_error)/(2*m)
    
    return J



def compute_cost_regularised_new(X, y, theta, alpha, l):
    
    """
    :param X        : 2D array of our dataset
    :param y        : 1D array of the groundtruth labels of the dataset
    :param theta    : 1D array of the trainable parameters
    :param l        : scalar, regularization parameter
    """
    
    # initialize costs
    # total_squared_error = 0.0
    # total_regularised_error = 0.0
    
    # get number of training examples
    m = y.shape[0]

    def ret_total_squared_error(X, y, theta):
        total_squared_error = 0.0
        m = y.shape[0]
        for i in range(m):
            hypothesis = calculate_hypothesis(X, theta, i)
            output = y[i]
            squared_error = (hypothesis - output)**2
            total_squared_error += squared_error
        return total_squared_error
    
    def ret_total_regularised_error(theta):
        total_regularised_error = 0.0
        for i in range(1,len(theta)):
            if i == 0:
                total_regularised_error += (theta[i] - ((alpha/m) * (ret_total_squared_error(X,y,theta) * X[0])))**2
            else:
                total_regularised_error += (theta[i]*(1 - alpha*(l/m)) - ((alpha)*(1/m) * (ret_total_squared_error(X,y,theta) * X[i])))**2
            # total_regularised_error += theta[i]**2
        return total_regularised_error
    
    total_squared_error = ret_total_squared_error(X, y, theta) 
    total_regularised_error = ret_total_regularised_error(theta)
    J = (total_squared_error + total_regularised_error)/(2*m)
    
    return J