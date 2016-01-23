from __future__ import print_function, division, unicode_literals
import random
import math
from toolz import take, pluck
import matplotlib.pyplot as pltl
from func_gradient_descent import gradient_descent
from pylsy2 import pylsytable2
from utility import until_within_tol
from out_utils import plot_lrates


def random_point(): 
    return (3 * random.random() - 1, 3 * random.random() - 1)
        
    
def g(x):
    """
    f(x, y) = -exp(-x^3 / 3 + x - y^2) has min at (1,0), saddle point at (-1,0)
    """
    return -math.exp(x[0]**3/-3 + x[0] - x[1]**2)
        

def dg(x):
    """just the gradient"""
    return ((1 - x[0]**2) * g(x), -2 * x[1] * g(x))
    
    
tol = 1.e-6
b = until_within_tol((g(e) for e in gradient_descent(dg, random_point())),
                     tolerance=tol)
print(list(b))

alphas = [1., 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
niter = 100

plot_lrates(g, dg, random_point(), alphas, niter)

#x0 = random_point()
x0 = [-0.2, 0.5]
result = list(take(50, ((g(e), e) for e in gradient_descent(dg, x0)) ))
xs = ['x' + unicode(i) for i in xrange(len(x0))]
table = pylsytable2(['y'] + xs)
table.add_data('y', list(pluck(0, result)), '{:.2e}')
for i, x in enumerate(xs):
    table.add_data(x, list(pluck(i,pluck(1, result))), '{:.2e}')
print(table)