from __future__ import print_function, division
import unittest
import random
import math
from utility import until_within_tol
from func_gradient_descent import gradient_descent


class FGDTest(unittest.TestCase):
    def test_sumsq(self):
        def f(x_i):
            return sum(x_ij**2 for x_ij in x_i)

        def df(x_i):
            return [2 * x_ij for x_ij in x_i]
        
        x0 = [5., 4.]
        tol = 1.e-6
        a = until_within_tol((f(e) for e in gradient_descent(df, x0)), tolerance=tol)
        b = list(a)
        self.assertLessEqual(abs(b[-2] - b[-1]), tol)

if __name__ == '__main__':
    unittest.main()