# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 11:39:25 2019

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt

def displayData(sel):
    fig, ax_array = plt.subplots(nrows=10, ncols=10, figsize=(10, 10))
    for row in range(10):
        for column in range(10):
            ax_array[row, column].matshow(sel[10 * row + column].reshape((20, 20)).T, cmap='gray')
            ax_array[row, column].axis('off')
    plt.show()
    return

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))

def rand_init_weight(L_in, L_out):
    w =  np.zeros((L_out, 1 + L_in))
    epsilon_init = 0.12
    w = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return w

def debug_init_weight(fan_out, fan_in):
    w = np.zeros((fan_out, 1 + fan_in))
    w = (np.sin(range(np.size(w))).reshape(w.shape)) / 10
    return w

def cost_function(nn_params, input_layer, hidden_layer, num_labels, X, y, lamb):
    theta1 = nn_params[0: (input_layer+1) * hidden_layer].reshape(hidden_layer, (input_layer + 1))
    theta2 = nn_params[(input_layer+1) * hidden_layer : ].reshape(num_labels, hidden_layer + 1)
    m = np.size(X, 0)
    J = 0
    
    K = num_labels
    a1 = np.vstack((np.ones((1,m)), X.T))
    z2 = theta1.dot(a1)
    a2 = np.vstack((np.ones((1,m)), sigmoid(z2)))
    z3 = theta2.dot(a2)
    a3 = sigmoid(z3).T
    Y = np.zeros((m,K))
    for i in range(m): 
        Y[i, int(y[i]-1)] = 1
    for i in range(m):
        for k in range(K):
            J = J + (-Y[i,k]) * np.log(a3[i,k]) - (1-Y[i,k]) * np.log(1-a3[i,k])
    J = J/m
    if lamb != 0:
        L = 0
        s1, t1 = theta1.shape
        for j in range(s1):
            for k in range(1,t1):
                L = L + np.power(theta1[j,k], 2)
        s2, t2 = theta2.shape
        for j in range(s2):
            for k in range(1, t2):
                L = L + np.power(theta2[j,k], 2)
        L = L * lamb/(2*m)
        J = J + L
    return J

def gradients(nn_params, input_layer, hidden_layer, num_labels, X, y, lamb):
    theta1 = nn_params[0: (input_layer+1) * hidden_layer].reshape(hidden_layer, (input_layer + 1))
    theta2 = nn_params[(input_layer+1) * hidden_layer : ].reshape(num_labels, hidden_layer + 1)
    m = np.size(X, 0)
    Y = np.zeros((m,num_labels))
    for i in range(m): 
        Y[i, int(y[i]-1)] = 1
    
    Delta1 = np.zeros(theta1.shape)
    Delta2 = np.zeros(theta2.shape)
    
    for i in range(m):
        a1 = np.vstack((1, X[i, :].T.reshape(len(X[i, :]), 1)))
        z2 = theta1.dot(a1)
        a2 = np.vstack((1, sigmoid(z2)))
        z3 = theta2.dot(a2)
        a3 = sigmoid(z3)
        delta3 = a3 - Y[i,:].T
        z2 = np.vstack((1, z2))
        delta2 = theta2.T.dot(delta3) * sigmoid_gradient(z2)
        
        Delta2 = Delta2 + delta3 * a2
        delta2 = delta2[1:]
        Delta1 = Delta1 + delta2 * a1.T
     
    Theta2_grad = Delta2/m
    Theta1_grad = Delta1/m
    
#    if lamb != 0:
#        Reg_delta1
    
    grad = np.hstack((Theta1_grad.flatten(), Theta2_grad.flatten()))
    return grad

def compute_numerical_gradient(J, theta):
    numgrad = np.zeros(theta.shape) 
    perturb = np.zeros(theta.shape)
    e = 1e-4
    for p in range(np.size(theta)):
        perturb[p] = e
        loss1 = J
        
def check_gradients(lamb = 0):
    input_layer = 3
    hidden_layer = 5
    num_labels = 3
    m = 5
    
    theta1 = debug_init_weight(hidden_layer, input_layer)
    theta2 = debug_init_weight(num_labels, hidden_layer)
    
    X = debug_init_weight(m, input_layer - 1)
    y = 1 + np.mod(np.arange(1,m+1), np.ones(m).dot(num_labels))
    
    nn_params = np.hstack((theta1.flatten(), theta2.flatten()))
    cost = cost_function(nn_params, input_layer, hidden_layer, num_labels, X, y, lamb)
    grad = gradients(nn_params, input_layer, hidden_layer, num_labels, X, y, lamb)
