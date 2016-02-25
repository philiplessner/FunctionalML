# coding: utf-8
from __future__ import print_function, division, unicode_literals
from toolz import pluck, compose
from tabulate import tabulate
from utility import csv_reader, Scaler, prepend_x0, dot
import logistic_regression as logr
import glm
from ml_util import train_test_split


# Get the iris data set
# SL: sepal length, SW: Sepal Width, PL: Petal Length, PW: Petal Width
# 0:  Iris Setosa 1: Iris Versicolour 2: Iris Virginica
Z, q = csv_reader('./data/iris.csv', ['SL', 'SW', 'PL', 'PW'], 'Type')
# Get Sepal Length and Petal Length features
Zp = list(pluck([0, 2], Z))
# Get only the Iris Setosa (0) and Iris Versicolour (1) classes
datap = [[f, o] for f, o in zip(Zp, q) if o != 2.0]
Xp, yp = zip(*datap)
y = list(yp)
Xpp = [list(e) for e in Xp]
print(Xpp)
print(y)

# Split set into training and testing data
train_data, test_data = train_test_split(zip(Xpp, y), 0.33)
# Scale the data
X_train, y_train = zip(*train_data)
scale = Scaler()
scale.fit(X_train)
transform = compose(prepend_x0, scale.transform)
scaledX_train = transform(X_train)
scaled_train = zip(scaledX_train, y_train)
# Fit the training data
h_theta0 = [1., 1., 1.]
print('****Gradient Descent****')
print('--Training--')
h_thetaf, cost = glm.fit(logr.logistic_log_likelihood,
                                logr.grad_logistic,
                                h_theta0,
                                scaled_train,
                                eta=0.03,
                                it_max=500,
                                gf='gd')

logr.plot_cost(cost)
print(h_thetaf)

yp_train = glm.predict(logr.logistic, scaledX_train, h_thetaf)
print(tabulate(list(zip(yp_train, map(round, yp_train), y_train)), 
                headers=['yp', 'Predicted Class', 'Actual Class'],
                tablefmt='fancy_grid'))

print('--Testing--')
# Use the training statistics to scale the test data
X_test, y_test = zip(*test_data)
scaledX_test = transform(X_test)
scaled_test = zip(scaledX_test, y_test)
yp_test = glm.predict(logr.logistic, scaledX_test, h_thetaf)
print(tabulate(list(zip(yp_test, map(round, yp_test), y_test)), 
                headers=['yp', 'Predicted Class', 'Actual Class'],
                tablefmt='fancy_grid'))

print('\n\n****Stochastic Gradient Descent****')
print('\n--Training--')
h_thetaf, cost = glm.fit(logr.logistic_log_likelihood_i,
                                logr.logistic_log_gradient_i,
                                h_theta0,
                                scaled_train,
                                eta=0.5,
                                it_max=500,
                                gf='sgd')
logr.plot_cost(cost)
print(h_thetaf)

yp_train = glm.predict(logr.logistic, scaledX_train, h_thetaf)
print(tabulate(list(zip(yp_train, map(round, yp_train), y_train)), 
                headers=['yp', 'Predicted Class', 'Actual Class'],
                tablefmt='fancy_grid'))

print('\n--Testing--')
# Use the training statistics to scale the test data
X_test, y_test = zip(*test_data)
scaledX_test = transform(X_test)
scaled_test = zip(scaledX_test, y_test)
yp_test = glm.predict(logr.logistic, scaledX_test, h_thetaf)
print(tabulate(list(zip(yp_test, map(round, yp_test), y_test)), 
                headers=['yp', 'Predicted Class', 'Actual Class'],
                tablefmt='fancy_grid'))