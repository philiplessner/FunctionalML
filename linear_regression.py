# coding: utf-8
from __future__ import print_function, division, unicode_literals
import random
from functools import partial
import matplotlib.pyplot as plt
from toolz import take, compose, curry
import func_gradient_descent as fgd
from utility import dot, until_within_tol, T, prepend_x0, vector_add


def error(xi, yi, h_theta):
    '''
    Difference between predicted and observed value for a training example
    Parameters
        xi: x vector (length j+1) for training example i
        yi: y observation for training example i
        h_theta: vector of parameters (theta0...thetaj)
    Returns
        error (predicted - observed)
    '''
    return dot(h_theta, xi) - yi


def errors(X, y, h_theta):
   return [error(xi, yi, h_theta) for (xi, yi) in zip(X, y)]
   

def J(X, y, h_theta):
    '''
    Cost function for multiple linear regression
    Parameters
        X: matrix of independent variables (i rows of observations and j cols of variables). x0=1 for all i
        y: dependent variable (i rows)
        h_theta: coefficients (j cols)
    Returns
        Cost function (sum of squares of errors)
    '''
    return sum(e**2 for e in errors(X, y, h_theta)) / (2. * len(y))


def gradJ(X, y, h_theta):
    '''
    Gradient of Cost function for batch gradient descent for
    Multiple linear regression
    Parameters
        X: matrix of independent variables (i rows of observations and j cols of variables). x0=1 for all i
        y: dependent variable (i rows)
        h_theta: coefficients (j cols)
    Returns
        Gradient of cost function (j cols, one for each h_thetaj)
        Will be used to update h_theta i gradient descent
    '''
    return [dot(errors(X, y, h_theta), xj) / len(y) for xj in T(X)]


def JS(xi, yi, h_theta):
    return 0.5 * error(xi, yi, h_theta)**2


def gradJS(xi, yi, h_theta):
    '''
    Gradient of Cost function for stochastic gradient descent for
    Multiple linear regression
    Uses a single observation to compute gradient
    Parameters
        xi: x vector (length j+1) for training example i
        yi: y observation for training example i
        h_theta: vector of parameters (theta0...thetaj) 
    Returns
        Gradient of cost function (j cols, one for each h_thetaj)
        Will be used to update h_theta in gradient descent
    '''
    return [(error(xi, yi, h_theta) * xj) for xj in xi]
    
    
def R(h_theta, alpha):
  return 0.5 * alpha * dot(h_theta[1:], h_theta[1:])


def ES(xi, yi, h_theta, alpha=0.0):
    """estimate error plus ridge penalty on h_theta"""
    return JS(xi, yi, h_theta) + R(h_theta, alpha)


def gradR(h_theta, alpha):
    """gradient of just the ridge penalty"""
    return [0] + [alpha * h_theta_j for h_theta_j in h_theta[1:]]


def gradES(x_i, y_i, h_theta, alpha=0.0):
    """the gradient corresponding to the ith squared error term
    including the ridge penalty"""
    return vector_add(gradJS(x_i, y_i, h_theta), gradR(h_theta, alpha))

    
@curry
def fit(cost_f, cost_df, h_theta0, data, eta=0.1, it_max=500, gf='gd'):
    '''
    Compute values of multiple linear regression coefficients
    Parameters
        cost_f: Cost function (J)
        cost_df: gradient of cost function (gradJ for batch and gradJS for stochastic)
        h_theta0: initial guess for fitting parameters (j cols)
        data: list of tuples [(Xi, yi)]
        X: matrix of independent variables (i rows of observations and j cols of variables). x0=1 for all i
        y: dependent variable (i rows)
        eta: learning rate
        it_max: maximum number of iterations
    Returns
        Fitting parameters (j cols)
    '''
    X, y = zip(*data)
    if gf == 'gd':
        f = partial(cost_f, X, y)
        df = partial(cost_df, X, y) 
        ans = list(take(it_max, 
                        ((h_theta, f(h_theta)) for h_theta in 
                          fgd.gradient_descent(df, h_theta0, eta=eta))))
        value = list(T(ans)[0])
        cost = list(T(ans)[1])
        #t = list(until_within_tol(cost, 1e-7))
        return value[-1], cost 
    elif gf == 'sgd':
        df = cost_df
        cost = [sum(cost_f(xi, yi, h_theta0) for xi, yi in data)]
        h_theta = h_theta0
        eta_new = eta
        for _ in xrange(it_max):
            ans = list(take(len(y), (e for e in fgd.sgd(df, X, y, h_theta, eta=eta_new))))
            h_theta = ans[-1]
            cost.append(sum(cost_f(xi, yi, h_theta) for xi, yi in data))
            eta_new = 0.99 * eta_new
        return h_theta, cost
    else:
        print('Not a valid function')
        return    
    
    
@curry
def predict(X, h_theta):
    return [dot(h_theta, xi) for xi in X]
    

def plot_cost(cost):
    plt.semilogy(range(0, len(cost)), cost, 'b+')
    plt.show()
    plt.clf()