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
    

def accuracy(tp, fp, fn, tn):
    correct = tp + tn
    total = tp + fp + fn + tn
    return correct / total


def precision(tp, fp, fn, tn):
    return tp / (tp + fp)


def recall(tp, fp, fn, tn):
    return tp / (tp + fn)


def f1_score(tp, fp, fn, tn):
    p = precision(tp, fp, fn, tn)
    r = recall(tp, fp, fn, tn)
    return 2 * p * r / (p + r)
