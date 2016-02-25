# coding: utf-8
from __future__ import print_function, division, unicode_literals
from math import exp, log
import matplotlib.pyplot as plt
from utility import dot, T


def logistic(z):
    return 1.0 / (1 + exp(-z))


def logistic_log_likelihood_i(x_i, y_i, h_theta):
    return y_i * log(logistic(dot(x_i, h_theta))) + (1 - y_i) * log(1 - logistic(dot(x_i, h_theta)))
       
    
def logistic_log_likelihood(X, y, h_theta):
    return sum(logistic_log_likelihood_i(x_i, y_i, h_theta) for x_i, y_i in zip(X, y))


def logistic_log_partial_ij(x_i, y_i, h_theta, j):
    """here i is the index of the data point,
    j the index of the derivative"""
    return (logistic(dot(x_i, h_theta)) - y_i) * x_i[j]
   
    
def logistic_log_gradient_i(x_i, y_i, h_theta):
    """the gradient of the log likelihood
    corresponding to the ith data point"""
    return [logistic_log_partial_ij(x_i, y_i, h_theta, j) for j, _ in enumerate(h_theta)]
    

def grad_logistic(X, y, h_theta):
    errors =[logistic(dot(h_theta, xi)) - yi for (xi, yi) in zip(X, y)]
    return [dot(errors, xj) for xj in T(X)]
    

def plot_cost(cost):
    plt.plot(range(0, len(cost)), cost, 'b+')
    plt.show()
    plt.clf()