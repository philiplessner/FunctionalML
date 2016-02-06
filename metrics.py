# coding: utf-8
from linear_regression import error
from utility import mean


def de_mean(x):
    """translate x by subtracting its mean (so the result has mean 0)"""
    xbar = mean(x)
    return [xi - xbar for xi in x]
    

def total_sum_of_squares(y):
    """the total squared variation of y_i's from their mean"""
    return sum(v ** 2 for v in de_mean(y))
    

def r2(X, y, h_theta):
    sum_of_squared_errors = sum(error(xi, yi, h_theta) ** 2
                                for xi, yi in zip(X, y))
    return 1.0 - sum_of_squared_errors / total_sum_of_squares(y)