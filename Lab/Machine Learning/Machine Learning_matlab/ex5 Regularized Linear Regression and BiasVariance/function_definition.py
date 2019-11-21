# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 17:12:25 2019

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from sklearn.preprocessing import StandardScaler
    
def cost_function(theta, X,  y,  lamb):
    m,n = X.shape    
    theta = theta.reshape((n,1))
    m = len(y)
    J = 0   
    j = sum(np.square(X.dot(theta) - y)) + sum(np.square(theta[1:,0]) * lamb)#theta[1:,0]
    J = j/(2*m)
    return J

def gradient(theta, X, y, lamb):
    m,n = X.shape      
    theta = theta.reshape((n,1))
    grad = np.zeros(theta.shape)
    deviation = X.dot(theta) - y
    grad[0,0] = np.matmul(X[:,0].T, deviation) / m
    grad[1:,:] =  (np.matmul(X[:,1:].T, deviation) + lamb*theta[1:,:])/m
    return grad

def train_linear_reg(X,y,lamb):
    initial_theta = np.zeros(np.size(X,1))
    result = op.minimize(fun=cost_function, x0=initial_theta, args=(X, y,lamb), method='TNC', jac=gradient)
    theta = result['x']
    return theta

def learning_curve(X, y, Xval, yval, lamb):
    m = np.size(X,0)
    error_train = list()
    error_val = list()
    for i in range(m):
        theta = train_linear_reg(X[0:i,:],y[0:i,:],lamb)
        error_t = cost_function(theta, X[0:i,:], y[0:i,:], 0)
        error_v = cost_function(theta, Xval, yval, 0)
        error_train.append(error_t)
        error_val.append(error_v)
    return error_train, error_val

# plot the fit over data
def plot_reg(X, y, theta):
    plt.figure(figsize = (12,8))
    plt.scatter(X[:,1], y[:,0],c = 'r', marker = 'x')
    plt.plot(X[:,1], X.dot(theta), 'b--')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()

def plot_learningcurve(error_train,error_val, m, titlestr):
    plt.figure(figsize = (12,8))
    plt.plot(range(m), error_train, label = 'Train', c='y')
    plt.plot(range(m), error_val, label = 'Cross Validation', c = 'k')
    plt.title(titlestr)
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

def plot_fit(min_x, max_x, mu, sigma, theta, p):
    x = np.arange(min_x - 15,max_x + 25, 0.05, float)#.reshape(2512,1)
    q = np.size(x,0)
    x_poly = poly_feature(x, p)
    x_norm = (x_poly - mu)/sigma
    x_poly = np.c_[np.ones((q, 1)), x_norm]
    plt.plot(x, x_poly.dot(theta), 'b--' )
    
def poly_feature(X, p):
    X_poly = np.zeros((np.size(X,0),p-1))
    for i in range(1,p):
        if X.shape != X_poly[:,1].shape:
            X = X.reshape(X_poly[:,i].shape)
        X_poly[:,i-1] = np.power(X,i)
    return X_poly

def feature_normalize(X):
    scaler = StandardScaler().fit(X)
    X_norm = scaler.transform(X)
    mu = scaler.mean_  #mu = np.mean(X_poly,axis = 0)
    sigma = np.sqrt(scaler.var_) #sigma = np.std(X_poly,axis = 0)
    return X_norm, mu, sigma

def validation_curve(X, y, Xval, yval): 
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).reshape(10,1)
    error_train = list()
    error_val = list()
    for i in range(len(lambda_vec)):
        lamb = lambda_vec[i,0]
        theta = train_linear_reg(X,y,lamb)
        error_t = cost_function(theta, X, y, 0)
        error_v = cost_function(theta, Xval, yval, 0)
        error_train.append(error_t)
        error_val.append(error_v)
    return lambda_vec, error_train, error_val     