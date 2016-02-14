# coding: utf-8
from __future__ import print_function, division, unicode_literals
from toolz import pluck, compose
from utility import csv_reader, Scaler, prepend_x0, dot
import logistic_regression as logr
from ml_util import train_test_split


# Get the iris data set
# SL: sepal length, SW: Sepal Width, PL: Petal Length, PW: Petal Width
# 0:  Iris Setosa 1: Iris Versicolour 2: Iris Virginica
Z, q = csv_reader('./data/iris.csv', ['SL', 'SW', 'PL', 'PW'], 'Type')
print(Z)
print(q)
# Get Sepal Length and Petal Length features
Zp = list(pluck([0, 2], Z))
# Get only the Iris Setosa (0) and Iris Versicolour (1) classes
datap = [[f, o] for f, o in zip(Zp, q) if o != 2.0]
Xp, yp = zip(*datap)
y = list(yp)
Xpp = [list(e) for e in Xp]
print(Xpp)
print(y)

# Scale the data
scale = Scaler(Xpp)
transform = compose(prepend_x0, Scaler.normalize)
scaledX = transform(scale)
# Split set into training and testing data
train_data, test_data = train_test_split(zip(scaledX, y), 0.33)
# Fit the training data
h_theta0 = [1., 1., 1.]
print('****Gradient Descent****')
print('--Training--')
h_thetaf, cost = logr.logistic_reg(logr.logistic_log_likelihood,
                                logr.grad_logistic,
                                h_theta0,
                                train_data,
                                it_max=500)

logr.plot_cost(cost)
print(h_thetaf)

for xi, yi in train_data:
    ypi = logr.logistic(dot(h_thetaf, xi))
    print(yi, ypi, round(ypi))

print('--Testing--')
for xi, yi in test_data:
    ypi = logr.logistic(dot(h_thetaf, xi))
    print(yi, ypi, round(ypi))

                                