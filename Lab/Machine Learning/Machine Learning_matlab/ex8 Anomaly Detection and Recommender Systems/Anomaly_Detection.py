# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:35:54 2019

@author: Modo
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import math

def load_data(filename):
    data = scipy.io.loadmat(filename)
    X = data['X']
    Xval = data['Xval']
    yval = data['yval']
    return X, Xval, yval

def plot_figure(X):
    plt.figure(figsize = (12,8))
    plt.scatter(X[:,0], X[:,1], c = 'b', marker = 'x')
    plt.axis([0,30,0,30])
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    #plt.show()

def estimate_gaussian(X):
    scaler = StandardScaler().fit(X)
    mu = scaler.mean_
    sigma2 = scaler.var_
    return mu, sigma2

def multivariate_gaussian(X, mu, sigma2):
    k = len(mu)
    sigma2 = sigma2.reshape(k,1)
    if (np.size(sigma2, 1) == 1) or (np.size(sigma2, 0) == 1):
        sigma2 = np.diag(sigma2[:,0])
    c = np.power(2 * math.pi, -k/2) * np.power(np.linalg.det(sigma2), -0.5)
    X = X-mu
    ans = X.dot(np.linalg.inv(sigma2)) * X
    po = -0.5 * ans.sum(axis = 1)
    p = c * np.exp(po.reshape(len(po),1))
    return p

def visualize_fit(x, mu, sigma2):
    X = np.arange(0, 35.5, 0.5)
    X1, X2 = np.meshgrid(X, X)
    Z = multivariate_gaussian(np.stack((X1.flatten(), X2.flatten()),  axis = -1), mu, sigma2)
    Z = Z.reshape(np.shape(X1))
    
    plt.scatter(x[:,0], x[:,1], c = 'b', marker = 'x')
    plt.contour(X1, X2, Z)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)') 
    plt.show()
      
def select_threshold(yval, pval):
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    
    start = pval.min()
    stop = pval.max()
    stepsize = (pval.max() - pval.min()) / 1000
    epsilons = np.arange(start, stop, stepsize)
    for epsilon in epsilons:
        predictions = pval < epsilon
        tp = sum((predictions == True) & (yval == 1))
        tp = sum((predictions == True) & (yval == 1))
        fp = sum((predictions == True) & (yval == 0))
        fn = sum((predictions == False) & (yval == 1))
        prec = tp/(tp+fp);
        rec = tp/(tp+fn);
        F1 = 2 * prec * rec / (prec + rec)

        if F1 > best_F1:
           best_F1 = F1
           best_epsilon = epsilon
    return best_epsilon, best_F1

if __name__ == "__main__":
    X, Xval, yval = load_data('ex8data1.mat')
    plot_figure(X)
    mu, sigma2 = estimate_gaussian(X)
    p = multivariate_gaussian(X, mu, sigma2)
    pval = multivariate_gaussian(Xval, mu, sigma2)
    visualize_fit(X,  mu, sigma2)
    epsilon, F1 = select_threshold(yval, pval)
    print('Best epsilon found using cross-validation: ', epsilon)
    print('Best F1 on Cross Validation Set:', F1)
    outlier = np.where(pval < epsilon)[0]
    plot_figure(X)
    plt.scatter(X[outlier][:,0], X[outlier][:,1], c = 'r', marker = 'o',)
    plt.show()
