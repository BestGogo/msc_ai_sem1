# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 16:23:03 2019

@author: Modo
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
import function_definition as fd

## Part2: ploting
data = np.loadtxt('ex1data1.txt', delimiter = ',')
X = data[:,0]
y = data[:,1]
m = len(y)
fd.plot_data(X,y)

## Part3: cosr and gradient descent
X1 = X.reshape(m,1)
y = y.reshape(m,1)
X = np.hstack((np.ones((m,1)),X1))
theta = np.zeros((2,1))
iterations = 1500
alpha = 0.01

print('\nTesting the cost function...')
J = fd.compute_cost(X, y, theta)
print('With theta = [0 ; 0],Cost computed = %f\n' %J)
print('Expected cost value (approx) 32.07\n')

J = fd.compute_cost(X, y, np.array([-1, 2]).reshape(2, 1))
print('With theta = [-1 ; 2],Cost computed = %f\n' %J)
print('Expected cost value (approx) 54.24\n')

print('\nRunning gradien descent...')
theta, J_history = fd.gradient_descent(X, y, theta, alpha, iterations)
print('Theta found by gradient descent:')
print(theta)
print('Expected theta values (approx)\n -3.6303\n  1.1664\n')

# plot the linear fit
fd.plot_data(X[:,1],y)
plt.plot(X[:,1],np.dot(X,theta), 'b-', linewidth = 1, label = 'Linear regression')
plt.legend()
plt.show()

#Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
print('For population = 35,000, we predict a profit of %f\n',predict1 * 10000)
predict2 = np.array([1, 7]).dot(theta)
print('For population = 70,000, we predict a profit of %f\n',predict2 * 10000)

## Part4: Visualizing J(theta_0, theta_1)
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([theta0_vals[i], theta1_vals[j]]).reshape(2,1)
        J_vals[i,j] = fd.compute_cost(X, y, t)
J_vals = J_vals.T
# Surface plot
fig = plt.figure(figsize = (12, 8))
ax = fig.gca(projection = '3d')
surf = ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap = cm.coolwarm,linewidth=0, antialiased=False) 
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()

#Contour plot 
plt.figure(figsize = (12, 8))
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20) )   
plt.scatter(theta[0][0], theta[1][0], c = 'r', marker = 'x')
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()


##using LinearRegression to calculate theta
from sklearn.linear_model import LinearRegression

data = np.loadtxt('ex1data1.txt', delimiter = ',')
X = data[:,0].reshape(-1,1)
y = data[:,1]
reg = LinearRegression().fit(X, y)
print(reg.intercept_)
print(reg.coef_)
