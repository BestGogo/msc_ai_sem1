# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:09:04 2019

@author: Administrator
"""

import numpy as np
import scipy.io as sc
import nn_function_definition as nnfd

## setup the parameters which will use for this exercise
input_layer = 400
hidden_layer = 25
num_labels = 10

## loading and visualizing data
data = sc.loadmat('ex4data1.mat')
X = data['X']
y = data['y']
m = np.size(X,0)

sel = np.random.permutation(m)
sel = X[sel[:100],:]
nnfd.displayData(sel)

## loading parameters
params = sc.loadmat('ex4weights.mat')
theta1 = params['Theta1']
theta2 = params['Theta2']
nn_params = np.hstack((theta1.flatten(), theta2.flatten()))

## compute cost (feedforward)
lamb = 0
J = nnfd.cost_function(nn_params, input_layer, hidden_layer, num_labels, X, y, lamb)
print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.287629)' % J)

## implement regularization
lamb = 1
J = nnfd.cost_function(nn_params, input_layer, hidden_layer, num_labels, X, y, lamb)
print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.383770)' % J)

## sigmoid gradient
g = nnfd.sigmoid_gradient(np.array([-1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n ')
print(g)

## initializing parameters
initial_theta1 = nnfd.rand_init_weight(input_layer, hidden_layer)
initial_theta2 = nnfd.rand_init_weight(hidden_layer, num_labels)
initial_nn_params = np.hstack((initial_theta1.flatten(), initial_theta2.flatten()))

## implement backpropagation
check_gradients()