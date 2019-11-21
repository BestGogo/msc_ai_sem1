# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 09:42:44 2019

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
def plot_data(x, y):
    plt.figure(figsize = (8,6))
    plt.scatter(x,y, color ='r', marker = 'x', label = 'Training data')
    plt.xlabel('Population of City in 10,000s')
    plt.ylabel('Profit in $10,000s')

def compute_cost(X, y, theta):
    m = len(y)
    diff = np.dot(X,theta) - y    
    J = np.square(diff).sum() / (2*m)
    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = list()
    #theta_hist = np.zeros((2,1))
    for i in range(num_iters):
        for j in range(len(theta)):
            theta[j,0] = theta[j,0] - (alpha / m) * np.matmul(np.transpose(np.dot(X,theta) - y), X[:,j])        
        #theta_hist = np.append(theta_hist,theta,axis = 0)
        J = compute_cost(X, y, theta)
        J_history.append(J)
    return theta, J_history

def feature_normalize(X):
    mu = np.mean(X, axis = 0)
    sigma = np.std(X,axis = 0)
    X_norm = (X-mu)/sigma
    return X_norm, mu, sigma

#def compute_cost_multi(X, y, theta):
#    m = len(y)
#    diff = np.dot(X,theta) - y    
#    J = np.matmul(diff.T, diff) / (2*m)
#    return J

def gradient_descent_multi(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = list()
    for i in range(num_iters):
        delta = np.transpose(np.matmul(np.transpose(np.matmul(X, theta) - y), X))
        theta = theta - alpha * delta / m   
        J = compute_cost(X, y, theta)
        J_history.append(J)
    return theta, J_history

def normal_eqn(X, y):
    theta = np.zeros((np.size(X, 1),1)) 
    X_inv = np.linalg.inv(np.matmul(X.T, X))
    theta =  np.matmul(np.matmul(X_inv, np.transpose(X)), y)
    return theta