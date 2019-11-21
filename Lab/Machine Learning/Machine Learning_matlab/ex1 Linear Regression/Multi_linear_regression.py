# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:09:46 2019

@author: Modo
"""

import numpy as np
import matplotlib.pyplot as plt
import function_definition as fd


## Part1: Feature Normalizetion
data = np.loadtxt('ex1data2.txt', delimiter = ',')
X = data[:, :-1]
y = data[:, -1]
m = len(y)
y = y.reshape(m,1)

#print('First 10 examples from the dataset: \n')
#print(np.c_[X[0:10,:], y[0:10, :]])

X, mu, sigma = fd.feature_normalize(X)
X = np.hstack((np.ones((m,1)),X))

alpha = 0.01
num_iters = 400
theta = np.zeros((np.size(X,1), 1))
theta, J_history = fd.gradient_descent_multi(X, y, theta, alpha, num_iters)
min_cost = np.min(J_history)
print("min cost",min_cost)
plt.figure(figsize = (12,8))
plt.plot(range(len(J_history)), J_history, 'b-', linewidth = 2)
plt.xlabel('Numbel ofiterations')
plt.ylabel('Cost J')

print('Theta computed from gradient descent: \n', theta)
print(100 * '-')

# Part3 Normal Equations
data = np.loadtxt('ex1data2.txt', delimiter = ',')
X = data[:, :-1]
y = data[:, -1]
m = len(y)
X = np.hstack((np.ones((m,1)), X))

theta  = fd.normal_eqn(X, y)    
print('Theta computed from the normal equations: \n', theta);

xp = np.array([1, 1650, 3]).reshape(1,3)
price = np.dot(xp, theta)
print(['Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n '], price)
print(100 * '-')

# sklearn linear regression
from sklearn.linear_model import LinearRegression 
data = np.loadtxt('ex1data2.txt', delimiter = ',')
X = data[:, :-1]
y = data[:, -1]

reg = LinearRegression()
reg.fit(X, y)
reg.score(X, y)
print('intercept:', reg.intercept_)
print('coef:', reg.coef_)
reg.predict(np.array([[1650, 3]]))


