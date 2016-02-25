# coding: utf-8
from __future__ import print_function, division, unicode_literals
import matplotlib.pyplot as plt
from toolz import take
from tabulate import tabulate
from func_gradient_descent import gradient_descent


def plot_lrates(f, df, x0, etas, niter):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    for eta in etas: 
        ax.plot(list(xrange(1, niter + 1)),
                    list(take(niter,(f(e) for e in gradient_descent(df, x0, eta=eta)))), 
                     label=unicode(eta))
    ax.set_xlabel('Iteration Number')
    ax.set_ylabel('f(x)')
    plt.legend(title='Learning Rate')
    plt.show()
    plt.clf()

    
def logistic_table(probs, yp, y):   
    print(tabulate(list(zip(probs, yp, y)), 
                headers=['Probability', 'Predicted Class', 'Actual Class'],
                tablefmt='fancy_grid'))