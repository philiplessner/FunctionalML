# coding: utf-8
from __future__ import print_function, division, unicode_literals
from functools import partial
from toolz import compose
from tabulate import tabulate
from utility import csv_reader, Scaler, prepend_x0
import metrics
import linear_regression as lr
from ml_util import train_test_split
import numpy as np
from numpy.linalg import lstsq

# Get the data
Z, y = csv_reader('./data/Folds_small.csv', ['AT', 'V', 'AP', 'RH'], 'PE') 
data = zip(Z, y)
# Split into a train set and test set
train_data, test_data = train_test_split(data, 0.33)
# Scale the training data
scale = Scaler()
Z_train, y_train = zip(*train_data)
scale.fit(Z_train)
transform = compose(prepend_x0, scale.transform)
X_train = transform(Z_train)
scaledtrain_data = zip(X_train, y_train)
# Scale the testing data using the same scaling parameters
# used for the training data
Z_test, y_test = zip(*test_data)
X_test = transform(Z_test)

h_theta0 = [0., 0., 0., 0., 0.]
print('****Gradient Descent****')
h_thetaf, cost = lr.fit(lr.J, 
                        lr.gradJ, 
                        h_theta0, 
                        eta=0.3, 
                        it_max=5000, gf='gd')(scaledtrain_data)
lr.plot_cost(cost)
h_thetad = scale.denormalize(h_thetaf)
yp_train = lr.predict(X_train, h_thetaf)

print('\n--Training--')
print('Coefficients\t', h_thetaf)
print(tabulate(list(zip(yp_train, y_train)), 
                headers=['yp', 'yi'],
                tablefmt='fancy_grid'))
    

print('Coefficients\t', h_thetad)
for i, h_theta in enumerate(h_thetad):
    print('h_theta' + unicode(i), '\t', h_theta)

corr_train = metrics.r2(X_train, y_train, h_thetaf)
print('R**2\t', corr_train)

print('\n--Testing--')
yp_test = lr.predict(X_test, h_thetaf)
print(tabulate(list(zip(yp_test, y_test)), 
                headers=['yp', 'yi'],
                tablefmt='fancy_grid'))
corr_test = metrics.r2(X_test, y_test, h_thetaf)
print('R**2\t', corr_test)

print('\n\n****Stochastic Gradient Descent****')
print('\n--Training--')
alpha = 0.0
h_thetaf, cost = lr.fit(partial(lr.ES, alpha=alpha),
                  partial(lr.gradES, alpha=alpha),
                  h_theta0, 
                  eta=0.1,
                  it_max=1000, gf='sgd')(scaledtrain_data)

lr.plot_cost(cost)
print('Coefficients\t', h_thetaf)
yp_train = lr.predict(X_train, h_thetaf)
h_thetad = scale.denormalize(h_thetaf)
print(tabulate(list(zip(yp_train, y_train)), 
                headers=['yp', 'yi'],
                tablefmt='fancy_grid'))
print('Coefficients\t', h_thetad)
for i, h_theta in enumerate(h_thetad):
    print('h_theta' + unicode(i), '\t', h_theta)

corr_train = metrics.r2(X_train, y_train, h_thetaf)
print('R**2\t', corr_train)

print('\n--Testing--')
yp_test = lr.predict(X_test, h_thetaf)
print(tabulate(list(zip(yp_test, y_test)), 
                headers=['yp', 'yi'],
                tablefmt='fancy_grid'))
corr_test = metrics.r2(X_test, y_test, h_thetaf)
print('R**2\t', corr_test)

print('****Numpy Solution****')
Q = np.array([e + [1.] for e in Z])
coeff = lstsq(Q, np.array(y))
print(coeff)
