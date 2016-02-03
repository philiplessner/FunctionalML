from __future__ import print_function, division, unicode_literals
import random
from functools import partial
from itertools import repeat, chain, cycle
from toolz import iterate, take, accumulate, curry
from utility import safe


@curry
def gradient_step(df, alpha, x_i):
    '''
    Calculate x_i+1 from x_i taking a step in negative direction of gradient
    x is a j dimensional vector [x1, x2,...,xj]
    Parameters
        df: Gradient of function f [df1, df2,...,dfj]
        alpha: step size
        x_i: [x_i1, x_i2,...,x_ij]
    Returns
       [x_i+11, x_i+12,...,x_i+1j] 
    '''
    return [x_ij - alpha * df_j for x_ij, df_j in zip(x_i, df(x_i))]
    

def gradient_descent(df, x_0, alpha=0.1):
    '''
    Parameters
        df: Gradient of function f
        x_0: Initial guess, x ia a j dimensional vector ([x_01, x_02,...,x_0j])
        alpha: Step size
    Returns
        Generator sequence of [x_i1, x_i2,...,x_ij] where i = 0 to ...
    '''
    return iterate(gradient_step(df, alpha), x_0)


def sgd_step(df, alpha, prev_theta, xy_i):
  """df is a function of x_i, y_i, theta"""
  x_i, y_i = xy_i
  gradient = df(x_i, y_i, prev_theta)
  return [theta_j - alpha * df_j
          for theta_j, df_j in zip(prev_theta, gradient)]


def sgd(df, X, y, theta0, alpha=0.1):
  xys = chain([theta0], cycle(zip(X, y)))
  return accumulate(partial(sgd_step, df, alpha), xys)
    
    
def gradient_descent2(f, df, x):
    while True:
        yield x
        x = min((partial(gradient_step, df, -alpha)(x) for alpha in [100, 10, 1, 0.7, 0.01, 0.001, 0.0001, 0.00001]), key=safe(f))


def gradient_descent3(f, df, x):
    return accumulate(lambda fx,_: min((partial(gradient_step, df, -alpha)(fx) for alpha in [100, 10, 1, 0.7, 0.01, 0.001, 0.0001, 0.00001]), key=safe(f)), repeat(x))