# coding: utf-8
from __future__ import print_function, division, unicode_literals
from functools import partial
from toolz import take, curry
import func_gradient_descent as fgd
from utility import dot, until_within_tol, T, vector_add


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
def predict(f, X, h_theta):
    return [f(dot(h_theta, xi)) for xi in X]
