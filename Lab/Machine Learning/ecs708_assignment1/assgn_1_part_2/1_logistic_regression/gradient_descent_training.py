from calculate_hypothesis import *
from compute_cost import *
from plot_cost_train_test import *

def gradient_descent_training(X_train, y_train, X_test, y_test, theta, alpha, iterations):
    """
        :param X_train      : 2D array of our training set
        :param y_train      : 1D array of the groundtruth labels of the training set
        :param X_test       : 2D array of our test set
        :param y_test       : 1D array of the groundtruth labels of the test set
        :param theta        : 1D array of the trainable parameters
        :param alpha        : scalar, learning rate
        :param iterations   : scalar, number of gradient descent iterations
    """
    
    m = X_train.shape[0] # the number of training samples is the number of rows of array X
    cost_vector_train = np.array([], dtype=np.float32) # empty array to store the train cost for every iteration
    cost_vector_test = np.array([], dtype=np.float32) # empty array to store the test cost for every iteration

    theta_test = theta.copy()
    theta_train = theta.copy()
    # Gradient Descent
    for it in range(iterations):        
        # initialize temporary theta, as a copy of the existing theta array
        theta_temp = theta_train.copy()
        # print(len(theta_temp))
        sigma = np.zeros((len(theta)))
        # print(sigma)
        for index in range(len(theta_temp)): 
            for i in range(m):
                hypothesis = calculate_hypothesis(X_train, theta_train, i)
                output = y_train[i]
                sigma[index] = sigma[index] + (hypothesis - output) * X_train[i, index]

            theta_temp[index] = theta_temp[index] - (alpha/m) * sigma[index]
        # copy theta_temp to theta
        theta_train = theta_temp.copy()
        
        # append current iteration's cost to cost_vector
        iteration_cost = compute_cost(X_train, y_train, theta_train)
        # print(iteration_cost)
        cost_vector_train = np.append(cost_vector_train, iteration_cost)
    # cost_vector_test = compute_cost(X_test, y_test, theta)
    m = X_test.shape[0]
    for it in range(iterations):        
        # initialize temporary theta, as a copy of the existing theta array
        theta_temp = theta_test.copy()
        # print(len(theta_temp))
        sigma = np.zeros((len(theta)))
        # print(sigma)
        for index in range(len(theta_temp)): 
            for i in range(m):
                hypothesis = calculate_hypothesis(X_test, theta_test, i)
                output = y_test[i]
                sigma[index] = sigma[index] + (hypothesis - output) * X_test[i, index]

            theta_temp[index] = theta_temp[index] - (alpha/m) * sigma[index]
        # copy theta_temp to theta
        theta_test = theta_temp.copy()
        
        # append current iteration's cost to cost_vector
        iteration_cost = compute_cost(X_test, y_test, theta_test)
        # print(iteration_cost)
        cost_vector_test = np.append(cost_vector_test, iteration_cost)
    print('Gradient descent finished.')
    
    return theta_train, cost_vector_train, cost_vector_test
