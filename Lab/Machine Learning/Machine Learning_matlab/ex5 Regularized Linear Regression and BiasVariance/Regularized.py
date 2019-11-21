# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 09:19:27 2019

@author: Administrator
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import prettytable
import function_definition as fd

## part1: Loading and Visualizing Data
data = loadmat('ex5data1.mat')
X = data['X']
y = data['y']
Xtest = data['Xtest']
ytest = data['ytest']
Xval = data['Xval']
yval = data['yval']

plt.figure(figsize = (12,8))
plt.scatter(X, y, c = 'r', marker = 'x')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

## Part2: regularized linear regression cost  
m = np.size(X,0)
X = np.hstack((np.ones((m, 1)), X)) 
theta = np.array([1,1])
J = fd.cost_function(theta, X ,  y,  1)
print('Cost at theta = [1 ; 1]: %f \n(this value should be about 303.993192)\n' %J)

## Part3: regularized linear regression gradient
grad = fd.gradient(theta, X, y, 1)
print('Gradient at theta = [1 ; 1] :\n(this value should be about [-15.303016; 598.250744])\n', grad)
    

## Part4: Train linear regression
lamb = 0
theta = fd.train_linear_reg(X,y,lamb) 
fd.plot_reg(X, y, theta)

## Part5: Learning curve for linear regression
lamb = 0
Xval = np.hstack((np.ones((len(Xval), 1)), Xval))
error_train, error_val = fd.learning_curve(X, y, Xval, yval, lamb)
titlestr = 'Learning curve for linear regression'
fd.plot_learningcurve(error_train,error_val, m, titlestr)

table = prettytable.PrettyTable()
table.add_column('Training example', np.arange(m, dtype = int))
table.add_column('Train Error', error_train)
table.add_column('Validation Error', error_val)
print(table)

## Part6: feature mapping for polynomial regression

p = 9
X = X[:, 1]
m = np.size(X,0)
X_poly = fd.poly_feature(X, p)
X_norm, mu, sigma = fd.feature_normalize(X_poly)
X_poly = np.c_[np.ones((m, 1)), X_norm]

n = np.size(Xtest,0)
X_poly_test = fd.poly_feature(Xtest, p)
X_norm_test = (X_poly_test - mu)/sigma
X_poly_test = np.c_[np.ones((n, 1)), X_norm_test]

o = np.size(Xval,0)
Xval = Xval[:, 1]
X_poly_val = fd.poly_feature(Xval, p)
X_norm_val = (X_poly_val - mu)/sigma
X_poly_val = np.c_[np.ones((o, 1)), X_norm_val]

print('Normalized Training Example 1:')
print(X_poly[0,:])

## Part7: learning curve for polynomial regression
lamb = 0
theta = fd.train_linear_reg(X_poly,y,lamb)
   
plt.figure(figsize = (12,8))
plt.scatter(X,y, c='r', marker = 'x')
fd.plot_fit(X.min(), X.max(), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title(('Polynomial Regression Fit (lambda = %f)' %lamb))
plt.xlim(-80, 80)
plt.ylim(-80,40)
    
plt.figure()
error_train, error_val = fd.learning_curve(X_poly, y, X_poly_val, yval, lamb)
titlestr = 'Polynomial Regression Learning Curve(lambda = %f)'%(lamb)
fd.plot_learningcurve(error_train,error_val, m, titlestr)

table = prettytable.PrettyTable()
table.add_column('Training example', np.arange(m, dtype = int))
table.add_column('Train Error', error_train)
table.add_column('Validation Error', error_val)
print(table)

## part8: validation for selecting lambda
lambda_vec, error_train, error_val  =  fd.validation_curve(X_poly, y, X_poly_val, yval)   
plt.figure(figsize = (12,8))
plt.plot(lambda_vec, error_train, label = 'Train', c='y')
plt.plot(lambda_vec, error_val, label = 'Cross Validation', c = 'k')
plt.xlabel('Number of training examples')
plt.ylabel('lambda')
plt.legend()
plt.show()  

table = prettytable.PrettyTable()
table.add_column('lambda',lambda_vec)
table.add_column('Train Error', error_train)
table.add_column('Validation Error', error_val)
print(table)