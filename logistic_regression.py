from __future__ import print_function, division, unicode_literals
import random
from math import exp, log
from functools import partial
from itertools import chain
import matplotlib.pyplot as plt
from toolz import take, thread_first, curry, compose
import func_gradient_descent as fgd
from ml_util import train_test_split
from utility import dot, until_within_tol, T, Scaler, prepend_x0


def logistic(z):
    return 1.0 / (1 + exp(-z))


def logistic_log_likelihood_i(x_i, y_i, h_theta):
    return y_i * log(logistic(dot(x_i, h_theta))) + (1 - y_i) * log(1 - logistic(dot(x_i, h_theta)))
       
    
def logistic_log_likelihood(X, y, h_theta):
    return sum(logistic_log_likelihood_i(x_i, y_i, h_theta) for x_i, y_i in zip(X, y))


def logistic_log_partial_ij(x_i, y_i, h_theta, j):
    """here i is the index of the data point,
    j the index of the derivative"""
    return (y_i - logistic(dot(x_i, h_theta))) * x_i[j]
   
    
def logistic_log_gradient_i(x_i, y_i, h_theta):
    """the gradient of the log likelihood
    corresponding to the ith data point"""
    return [logistic_log_partial_ij(x_i, y_i, h_theta, j) for j, _ in enumerate(h_theta)]
    

def grad_logistic(X, y, h_theta):
    errors =[logistic(dot(h_theta, xi)) - yi for (xi, yi) in zip(X, y)]
    return [dot(errors, xj) for xj in T(X)]
    
@curry    
def logistic_reg(cost_f, cost_df, h_theta0, data, it_max=500):
    X, y = zip(*data) 
    f = partial(cost_f, X, y)
    df = partial(cost_df, X, y)
    ans = list(take(it_max, ((e, f(e)) for e in fgd.gradient_descent(df, h_theta0, eta=0.03))))
    value = list(T(ans)[0])
    cost = list(T(ans)[1])
    #t = list(until_within_tol(cost, 1e-7))
    return value[-1], cost 


@curry
def predict(f, X, h_theta):
    return [f(dot(h_theta, xi)) for xi in X]
    

def plot_cost(cost):
    plt.plot(range(0, len(cost)), cost, 'b+')
    plt.show()
    plt.clf()
    

if __name__ == "__main__":

    data = [(0.7,48000,1),(1.9,48000,0),(2.5,60000,1),(4.2,63000,0),(6,76000,0),(6.5,69000,0),(7.5,76000,0),(8.1,88000,0),(8.7,83000,1),(10,83000,1),(0.8,43000,0),(1.8,60000,0),(10,79000,1),(6.1,76000,0),(1.4,50000,0),(9.1,92000,0),(5.8,75000,0),(5.2,69000,0),(1,56000,0),(6,67000,0),(4.9,74000,0),(6.4,63000,1),(6.2,82000,0),(3.3,58000,0),(9.3,90000,1),(5.5,57000,1),(9.1,102000,0),(2.4,54000,0),(8.2,65000,1),(5.3,82000,0),(9.8,107000,0),(1.8,64000,0),(0.6,46000,1),(0.8,48000,0),(8.6,84000,1),(0.6,45000,0),(0.5,30000,1),(7.3,89000,0),(2.5,48000,1),(5.6,76000,0),(7.4,77000,0),(2.7,56000,0),(0.7,48000,0),(1.2,42000,0),(0.2,32000,1),(4.7,56000,1),(2.8,44000,1),(7.6,78000,0),(1.1,63000,0),(8,79000,1),(2.7,56000,0),(6,52000,1),(4.6,56000,0),(2.5,51000,0),(5.7,71000,0),(2.9,65000,0),(1.1,33000,1),(3,62000,0),(4,71000,0),(2.4,61000,0),(7.5,75000,0),(9.7,81000,1),(3.2,62000,0),(7.9,88000,0),(4.7,44000,1),(2.5,55000,0),(1.6,41000,0),(6.7,64000,1),(6.9,66000,1),(7.9,78000,1),(8.1,102000,0),(5.3,48000,1),(8.5,66000,1),(0.2,56000,0),(6,69000,0),(7.5,77000,0),(8,86000,0),(4.4,68000,0),(4.9,75000,0),(1.5,60000,0),(2.2,50000,0),(3.4,49000,1),(4.2,70000,0),(7.7,98000,0),(8.2,85000,0),(5.4,88000,0),(0.1,46000,0),(1.5,37000,0),(6.3,86000,0),(3.7,57000,0),(8.4,85000,0),(2,42000,0),(5.8,69000,1),(2.7,64000,0),(3.1,63000,0),(1.9,48000,0),(10,72000,1),(0.2,45000,0),(8.6,95000,0),(1.5,64000,0),(9.8,95000,0),(5.3,65000,0),(7.5,80000,0),(9.9,91000,0),(9.7,50000,1),(2.8,68000,0),(3.6,58000,0),(3.9,74000,0),(4.4,76000,0),(2.5,49000,0),(7.2,81000,0),(5.2,60000,1),(2.4,62000,0),(8.9,94000,0),(2.4,63000,0),(6.8,69000,1),(6.5,77000,0),(7,86000,0),(9.4,94000,0),(7.8,72000,1),(0.2,53000,0),(10,97000,0),(5.5,65000,0),(7.7,71000,1),(8.1,66000,1),(9.8,91000,0),(8,84000,0),(2.7,55000,0),(2.8,62000,0),(9.4,79000,0),(2.5,57000,0),(7.4,70000,1),(2.1,47000,0),(5.3,62000,1),(6.3,79000,0),(6.8,58000,1),(5.7,80000,0),(2.2,61000,0),(4.8,62000,0),(3.7,64000,0),(4.1,85000,0),(2.3,51000,0),(3.5,58000,0),(0.9,43000,0),(0.9,54000,0),(4.5,74000,0),(6.5,55000,1),(4.1,41000,1),(7.1,73000,0),(1.1,66000,0),(9.1,81000,1),(8,69000,1),(7.3,72000,1),(3.3,50000,0),(3.9,58000,0),(2.6,49000,0),(1.6,78000,0),(0.7,56000,0),(2.1,36000,1),(7.5,90000,0),(4.8,59000,1),(8.9,95000,0),(6.2,72000,0),(6.3,63000,0),(9.1,100000,0),(7.3,61000,1),(5.6,74000,0),(0.5,66000,0),(1.1,59000,0),(5.1,61000,0),(6.2,70000,0),(6.6,56000,1),(6.3,76000,0),(6.5,78000,0),(5.1,59000,0),(9.5,74000,1),(4.5,64000,0),(2,54000,0),(1,52000,0),(4,69000,0),(6.5,76000,0),(3,60000,0),(4.5,63000,0),(7.8,70000,0),(3.9,60000,1),(0.8,51000,0),(4.2,78000,0),(1.1,54000,0),(6.2,60000,0),(2.9,59000,0),(2.1,52000,0),(8.2,87000,0),(4.8,73000,0),(2.2,42000,1),(9.1,98000,0),(6.5,84000,0),(6.9,73000,0),(5.1,72000,0),(9.1,69000,1),(9.8,79000,1),]
    data = map(list, data) # change tuples to lists
    # each element is [1, experience, salary]
    Z = [row[:2] for row in data]
    # each element is paid_account
    y = [row[2] for row in data]        
    scale = Scaler(Z)
    transform = compose(prepend_x0, Scaler.normalize)
    X = transform(scale)
    data = zip(X, y) 
    train_data, test_data = train_test_split(data, 0.33)
    h_theta0 = [1., 1., 1.]
    h_thetaf, cost = logistic_reg(logistic_log_likelihood, grad_logistic, h_theta0, train_data, it_max=5000)
    print(h_thetaf)
    h_thetad = scale.denormalize(h_thetaf)
    print(h_thetad)
    plot_cost(cost)
    true_positives = false_positives = true_negatives = false_negatives = 0
    for x_i, y_i in test_data:
        predict = logistic(dot(h_thetaf, x_i))
        if y_i == 1 and predict >= 0.5:  # TP: paid and we predict paid
            true_positives += 1
        elif y_i == 1:                   # FN: paid and we predict unpaid
            false_negatives += 1
        elif predict >= 0.5:             # FP: unpaid and we predict paid
            false_positives += 1
        else:                             #  TN: unpaid and we predict unpaid
            true_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    print('Precision: ', precision)
    print('Recall: ', recall)
    