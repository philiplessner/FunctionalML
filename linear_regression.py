# coding: utf-8
from __future__ import print_function, division, unicode_literals
import random
from functools import partial
import matplotlib.pyplot as plt
from toolz import take, compose, curry
import func_gradient_descent as fgd
from utility import dot, until_within_tol, T, csv_reader, Scaler, prepend_x0


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


def gradJS(xi, yi, h_theta):
    '''
    Gradient of Cost function for stochastic gradient descent for
    Multiple linear regression
    Uses a single observation to compute gradient
    Parameters
        X: matrix of independent variables (i rows of observations and j cols of variables). x0=1 for all i
        y: dependent variable (i rows)
        h_theta: coefficients (j cols) 
    Returns
        Gradient of cost function (j cols, one for each h_thetaj)
        Will be used to update h_theta i gradient descent
    '''
    #xi, yi = random.choice(zip(X,y))
    return [(error(xi, yi, h_theta) * xj) for xj in xi]
    
@curry
def fit(cost_f, cost_df, h_theta0, data, alpha=0.1, it_max=500, gf='gd'):
    '''
    Compute values of multiple linear regression coefficients
    Parameters
        cost_f: Cost function (J)
        cost_df: gradient of cost function (gradJ for batch and gradJS for stochastic)
        X: matrix of independent variables (i rows of observations and j cols of variables). x0=1 for all i
        y: dependent variable (i rows)
        h_theta0: initial guess for fitting parameters (j cols)
        it_max: maximum number of iterations
    Returns
        Fitting parameters (j cols)
    '''
    X, y = zip(*data)
    if gf == 'gd':
        f = partial(cost_f, X, y)
        df = partial(cost_df, X, y) 
        ans = list(take(it_max, ((e, f(e)) for e in fgd.gradient_descent(df, h_theta0, alpha=alpha))))
    elif gf == 'sgd':
        #f = partial(cost_f, X, y)
        df = cost_df 
        ans = list(take(it_max, (e for e in fgd.sgd(df, X, y, h_theta0, alpha=alpha))))
        return ans[-1]
    else:
        print('Not a valid function')
        return    
    value = list(T(ans)[0])
    cost = list(T(ans)[1])
    #t = list(until_within_tol(cost, 1e-7))
    return value[-1], cost
    
@curry
def predict(X, h_theta):
    return [dot(h_theta, xi) for xi in X]
    

def plot_cost(cost):
    plt.semilogy(range(0, len(cost)), cost, 'b+')
    plt.show()
    plt.clf()