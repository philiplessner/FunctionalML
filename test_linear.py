from __future__ import print_function, division, unicode_literals
import unittest
import random
from itertools import chain
from toolz import take, compose, curry
from toolz.curried import do, get
import func_gradient_descent as fgd
from linear_regression import predict, prepend_x0, lin_reg, J, gradJ
from utility import Scaler


class LinearTest(unittest.TestCase):
    def setUp(self):
        self.ferror = 0.05
        
    def test_synthetic(self):
        jvars = 2
        isamples = 100
        h_theta = [3.2, 5.5, 4.3] 
        Z = [[random.random() for _ in range(jvars)] for _ in range(isamples)]
        y = predict(prepend_x0(Z), h_theta)
        h_thetad = self._common(Z, y)
        h_thetaa = [round(x, 1) for x in h_thetad]
        self.assertListEqual(h_theta, h_thetaa)
        
    def test_slr(self):
        h_theta = [22.95, 0.903]
        Z = [49,41,40,25,21,21,19,19,18,18,16,15,15,15,15,14,14,13,13,13,13,12,12,11,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,8,8,8,8,8,8,8,8,8,8,8,8,8,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        y = [68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]
        Q = [[e] for e in Z]
        h_thetad = self._common(Q, y)
        fdiff = self._list_absfdiff(h_theta, h_thetad)
        for e in fdiff:
            self.assertLessEqual(e, self.ferror)
        
    def test_mlr(self):
        h_theta = [30.63, 0.972, -1.868, 0.911] 
        Z = [[49, 4, 0], [41, 9, 0], [40, 8, 0], [25, 6, 0], [21, 1, 0], [21, 0, 0], [19, 3, 0], [19, 0, 0], [18, 9, 0], [18, 8, 0], [16, 4, 0], [15, 3, 0], [15, 0, 0], [15, 2, 0], [15, 7, 0], [14, 0, 0], [14, 1, 0], [13, 1, 0], [13, 7, 0], [13, 4, 0], [13, 2, 0], [12, 5, 0], [12, 0, 0], [11, 9, 0], [10, 9, 0], [10, 1, 0], [10, 1, 0], [10, 7, 0], [10, 9, 0], [10, 1, 0], [10, 6, 0], [10, 6, 0], [10, 8, 0], [10, 10, 0], [10, 6, 0], [10, 0, 0], [10, 5, 0], [10, 3, 0], [10, 4, 0], [9, 9, 0], [9, 9, 0], [9, 0, 0], [9, 0, 0], [9, 6, 0], [9, 10, 0], [9, 8, 0], [9, 5, 0], [9, 2, 0], [9, 9, 0], [9, 10, 0], [9, 7, 0], [9, 2, 0], [9, 0, 0], [9, 4, 0], [9, 6, 0], [9, 4, 0], [9, 7, 0], [8, 3, 0], [8, 2, 0], [8, 4, 0], [8, 9, 0], [8, 2, 0], [8, 3, 0], [8, 5, 0], [8, 8, 0], [8, 0, 0], [8, 9, 0], [8, 10, 0], [8, 5, 0], [8, 5, 0], [7, 5, 0], [7, 5, 0], [7, 0, 0], [7, 2, 0], [7, 8, 0], [7, 10, 0], [7, 5, 0], [7, 3, 0], [7, 3, 0], [7, 6, 0], [7, 7, 0], [7, 7, 0], [7, 9, 0], [7, 3, 0], [7, 8, 0], [6, 4, 0], [6, 6, 0], [6, 4, 0], [6, 9, 0], [6, 0, 0], [6, 1, 0], [6, 4, 0], [6, 1, 0], [6, 0, 0], [6, 7, 0], [6, 0, 0], [6, 8, 0], [6, 4, 0], [6, 2, 1], [6, 1, 1], [6, 3, 1], [6, 6, 1], [6, 4, 1], [6, 4, 1], [6, 1, 1], [6, 3, 1], [6, 4, 1], [5, 1, 1], [5, 9, 1], [5, 4, 1], [5, 6, 1], [5, 4, 1], [5, 4, 1], [5, 10, 1], [5, 5, 1], [5, 2, 1], [5, 4, 1], [5, 4, 1], [5, 9, 1], [5, 3, 1], [5, 10, 1], [5, 2, 1], [5, 2, 1], [5, 9, 1], [4, 8, 1], [4, 6, 1], [4, 0, 1], [4, 10, 1], [4, 5, 1], [4, 10, 1], [4, 9, 1], [4, 1, 1], [4, 4, 1], [4, 4, 1], [4, 0, 1], [4, 3, 1], [4, 1, 1], [4, 3, 1], [4, 2, 1], [4, 4, 1], [4, 4, 1], [4, 8, 1], [4, 2, 1], [4, 4, 1], [3, 2, 1], [3, 6, 1], [3, 4, 1], [3, 7, 1], [3, 4, 1], [3, 1, 1], [3, 10, 1], [3, 3, 1], [3, 4, 1], [3, 7, 1], [3, 5, 1], [3, 6, 1], [3, 1, 1], [3, 6, 1], [3, 10, 1], [3, 2, 1], [3, 4, 1], [3, 2, 1], [3, 1, 1], [3, 5, 1], [2, 4, 1], [2, 2, 1], [2, 8, 1], [2, 3, 1], [2, 1, 1], [2, 9, 1], [2, 10, 1], [2, 9, 1], [2, 4, 1], [2, 5, 1], [2, 0, 1], [2, 9, 1], [2, 9, 1], [2, 0, 1], [2, 1, 1], [2, 1, 1], [2, 4, 1], [1, 0, 1], [1, 2, 1], [1, 2, 1], [1, 5, 1], [1, 3, 1], [1, 10, 1], [1, 6, 1], [1, 0, 1], [1, 8, 1], [1, 6, 1], [1, 4, 1], [1, 9, 1], [1, 9, 1], [1, 4, 1], [1, 2, 1], [1, 9, 1], [1, 0, 1], [1, 8, 1], [1, 6, 1], [1, 1, 1], [1, 1, 1], [1, 5, 1]]
        
        y = [68.77,51.25,52.08,38.36,44.54,57.13,51.4,41.42,31.22,34.76,54.01,38.79,47.59,49.1,27.66,41.03,36.73,48.65,28.12,46.62,35.57,32.98,35,26.07,23.77,39.73,40.57,31.65,31.21,36.32,20.45,21.93,26.02,27.34,23.49,46.94,30.5,33.8,24.23,21.4,27.94,32.24,40.57,25.07,19.42,22.39,18.42,46.96,23.72,26.41,26.97,36.76,40.32,35.02,29.47,30.2,31,38.11,38.18,36.31,21.03,30.86,36.07,28.66,29.08,37.28,15.28,24.17,22.31,30.17,25.53,19.85,35.37,44.6,17.23,13.47,26.33,35.02,32.09,24.81,19.33,28.77,24.26,31.98,25.73,24.86,16.28,34.51,15.23,39.72,40.8,26.06,35.76,34.76,16.13,44.04,18.03,19.65,32.62,35.59,39.43,14.18,35.24,40.13,41.82,35.45,36.07,43.67,24.61,20.9,21.9,18.79,27.61,27.21,26.61,29.77,20.59,27.53,13.82,33.2,25,33.1,36.65,18.63,14.87,22.2,36.81,25.53,24.62,26.25,18.21,28.08,19.42,29.79,32.8,35.99,28.32,27.79,35.88,29.06,36.28,14.1,36.63,37.49,26.9,18.58,38.48,24.48,18.95,33.55,14.24,29.04,32.51,25.63,22.22,19,32.73,15.16,13.9,27.2,32.01,29.27,33,13.74,20.42,27.32,18.23,35.35,28.48,9.08,24.62,20.12,35.26,19.92,31.02,16.49,12.16,30.7,31.22,34.65,13.13,27.51,33.2,31.57,14.1,33.42,17.44,10.12,24.42,9.82,23.39,30.93,15.03,21.67,31.09,33.29,22.61,26.89,23.48,8.38,27.81,32.35,23.84]
        h_thetad = self._common(Z, y)
        fdiff = self._list_absfdiff(h_theta, h_thetad)
        for e in fdiff:
            self.assertLessEqual(e, self.ferror) 
     
    def _common(self, Z, y):
        scale = Scaler(Z)
        transform = compose(prepend_x0, Scaler.normalize)
        X = transform(scale)
        data = zip(X, y)
        h_theta0 = [0.] * len(X[0])
        coeff = compose(scale.denormalize, 
                        get(0), 
                        lin_reg(J, gradJ, h_theta0, it_max=2000))
        h_thetad = coeff(data)
        return h_thetad
        
    def _list_absfdiff(self, l1, l2):
        return [abs((e1 - e2) / e1) for e1, e2 in zip(l1, l2)]
    

if __name__ == '__main__':
    unittest.main()