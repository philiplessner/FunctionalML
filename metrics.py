# coding: utf-8
from __future__ import division
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


class Scores(object):
    def __init__(self, y, yp):
        true_positives = false_positives = true_negatives = false_negatives = 0
        for yi, ypi in zip(y, yp):
            if yi == 1 and ypi == 1:
                true_positives += 1
            elif yi == 1 and ypi == 0:
                false_negatives += 1
            elif yi == 0 and ypi == 1:
                false_positives += 1
            else:
                true_negatives += 1
        self.tp = true_positives
        self.fn = false_negatives
        self.fp = false_positives
        self.tn = true_negatives
    
    def accuracy(self):
        correct = self.tp + self.tn
        total = self.tp + self.fp + self.fn + self.tn
        return correct / total
    
    def precision(self):
        return self.tp / (self.tp + self.fp)
        
    def recall(self):
        return self.tp / (self.tp + self.fn)
        
    def f1_score(self):
       return 2 * self.precision() * self.recall() / (self.precision() + self.recall()) 
