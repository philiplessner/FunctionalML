# coding: utf-8
from __future__ import print_function, division, unicode_literals
from toolz import compose
from tabulate import tabulate
from utility import csv_reader, Scaler, prepend_x0
import metrics
import linear_regression as lr
from ml_util import train_test_split
import numpy as np
from numpy.linalg import lstsq


Z, y = csv_reader('./data/Folds_small.csv', ['AT', 'V', 'AP', 'RH'], 'PE') 
    
scale = Scaler(Z)
transform = compose(prepend_x0, Scaler.normalize)
X = transform(scale)

data = zip(X, y)
train_data, test_data = train_test_split(data, 0.33)
h_theta0 = [0., 0., 0., 0., 0.]

h_thetaf, cost = lr.fit(lr.J, 
                        lr.gradJ, 
                        h_theta0, 
                        alpha=0.3, 
                        it_max=5000)(train_data)
lr.plot_cost(cost)
h_thetad = scale.denormalize(h_thetaf)
X_train, y_train = zip(*train_data)
X_test, y_test = zip(*test_data)
yp_train = lr.predict(X_train, h_thetaf)

print('****Training****')
print('Coefficients\t', h_thetaf)
print(tabulate(list(zip(yp_train, y_train)), 
                headers=['yp', 'yi'],
                tablefmt='fancy_grid'))
    

print('******************')
print('Coefficients\t', h_thetad)
for i, h_theta in enumerate(h_thetad):
    print('h_theta' + unicode(i), '\t', h_theta)

corr_train = metrics.r2(X_train, y_train, h_thetaf)
print('R**2\t', corr_train)

print('****Testing****')

yp_test = lr.predict(X_test, h_thetaf)
print(tabulate(list(zip(yp_test, y_test)), 
                headers=['yp', 'yi'],
                tablefmt='fancy_grid'))
corr_test = metrics.r2(X_test, y_test, h_thetaf)
print('R**2\t', corr_test)

Q = np.array([e + [1.] for e in Z])
coeff = lstsq(Q, np.array(y))
print(coeff)