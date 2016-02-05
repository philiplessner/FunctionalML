from __future__ import print_function, division, unicode_literals
import random
from functools import partial
from itertools import repeat, chain, cycle
from toolz import iterate, take, accumulate, curry
from utility import safe


@curry
def gradient_step(df, eta, theta_k):
    '''
    Calculate theta_k+1 from theta_k
    by taking step in negative direction of gradient 
    theta is a j dimensional vector
    Parameters
        df: Gradient of function f [df1, df2,...,dfj]
        eta: Learning rate
        theta_k: [theta_k1, theta_k2,...,theta_kj]
    Returns
       [theta_k+11, theta_k+12,...,thetak_k+1j] 
    '''
    return [theta_k - eta * df_k 
                                for theta_k, df_k in zip(theta_k, df(theta_k))]
    

def gradient_descent(df, theta_0, eta=0.1):
    '''
    Parameters
        df: Gradient of function f
        theta0: Initial guess, theta ia a j dimensional vector ([theta_01, theta_02,...,theta0_0j])
        eta: Learning rate
    Returns
        Generator sequence of [theta_k1, theta_k2,...,theta_kj] 
        where k = 0 to ...
    '''
    return iterate(gradient_step(df, eta), theta_0)


def sgd_step(df, eta, theta_k, xy_i):
    '''
    df is a function of x_i, y_i, theta
    '''
    x_i, y_i = xy_i
    gradient = df(x_i, y_i, theta_k)
    return [theta_k - eta * df_k
            for theta_k, df_k in zip(theta_k, gradient)]


def sgd(df, X, y, theta_0, eta=0.1):
    '''
    Parameters
        df: Gradient of function f
        X: Matrix of features
        y: vector of observations
        theta0: Initial guess, theta ia a j dimensional vector ([theta_01, theta_02,...,theta0_0j])
        eta: Learning rate
    Returns
        Generator sequence of [theta_k1, theta_k2,...,theta_kj] 
        where k = 0 to ...
    ''' 
    xys = chain([theta_0], cycle(zip(X, y)))
    return accumulate(partial(sgd_step, df, eta), xys)
    
    
def gradient_descent2(f, df, x):
    while True:
        yield x
        x = min((partial(gradient_step, df, -alpha)(x) for alpha in [100, 10, 1, 0.7, 0.01, 0.001, 0.0001, 0.00001]), key=safe(f))


def gradient_descent3(f, df, x):
    return accumulate(lambda fx,_: min((partial(gradient_step, df, -alpha)(fx) for alpha in [100, 10, 1, 0.7, 0.01, 0.001, 0.0001, 0.00001]), key=safe(f)), repeat(x))