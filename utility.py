from __future__ import print_function, division, unicode_literals
import csv
import random
from functools import partial
from math import sqrt
from toolz import accumulate, iterate, take, curry


@curry
def until_nearly_convergence(convf, it, tolerance=0.0001):
    '''
    Test for absolute convergence
    Parameters
        it: Lazy sequence of values
        tolerance: Convergence criteria
    Returns
        Continues to add to the sequence of current values if tolerence is not satisfied
        Othewise it terminates iteration and returns the sequence of values
    '''
    # The order of arguments for toolz.accumulate is opposite to
    # Python 3 itertools.accumulate
    return accumulate(partial(convf, tolerance), it)


def within_tolerance(tol, prev, curr):
    if abs(prev - curr) < tol:
        raise StopIteration                  
    else:
        return curr

def make_within_tolerance():
    d = {'stop': False}
    def within_tolerance2(tol, prev, curr):
        if d['stop'] is True:
            d['stop'] = False
            raise StopIteration
        elif abs(prev - curr) <= tol: 
            d['stop'] = True 
            return curr
        else: 
            return curr
    return within_tolerance2

until_within_tol = until_nearly_convergence(make_within_tolerance())

        
def within_ftolerance(ftol, prev, curr):
    if abs(prev - curr) < (curr * ftol): 
        raise StopIteration
    else: 
        return curr 


#until_within_tol = partial(until_nearly_convergence, within_tolerance)


until_within_ftol = partial(until_nearly_convergence, within_ftolerance)


def safe(f):
    '''
    return a new function that's the same as f,
    except that it outputs infinity whenever f produces an error
    '''
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
           # this means "infinity" in Python  
            return float('inf')         
    return safe_f


def partial_difference_quotient(f, v, i, h):
    """compute the ith partial difference quotient of f at v"""
   # add h to just the ith element of v 
    w = [v_j + (h if j == i else 0) for j, v_j in enumerate(v)]
    return (f(w) - f(v)) / h


def estimate_gradient(f, v, h=0.00001):
    return [partial_difference_quotient(f, v, i, h) for i, _ in enumerate(v)]
    

def divide_by_n(n, h0):
    '''
    Repetedly divide an initial valve by n
    Parameters
        n: Number to divide by
        h0: Initial value
    Returns
        (Possibly) infinite sequence h0, h0/n, h0/n**2,...
    '''
    return iterate(lambda x: x/n, h0)


halve = partial(divide_by_n, 2.)
thirds = partial(divide_by_n, 3.)
tenths = partial(divide_by_n, 10.)


def dot(a, b):
    '''
    Dot product of two vectors
    '''
    assert len(a) == len(b)
    return sum(ai * bj for (ai, bj) in zip(a, b))


def vector_add(v, w):
    """adds two vectors componentwise"""
    return [v_i + w_i for v_i, w_i in zip(v, w)]
    

def mean(x):
    '''
    Mean
    Parameters
        x: list of numbers
    Returns
        mean value of list
    '''
    return sum(x) / len(x)
    

def sdev(x):
    return sqrt(sum((xi - mean(x))**2 for xi in x))/(len(x) - 1)


def T(A):
    '''
    Transpose of a matrix
    '''
    return map(lambda *a: list(a), *A)


def csv_reader(fpath, Xcols, ycol):
    with open(fpath, 'rU') as f:
        X = []
        y = []   
        reader = csv.DictReader(f)
        for row in reader:
            X.append([float(row[col]) for col in Xcols])
            y.append(float(row[ycol]))
    return X, y


class Scaler(object):
    def __init__(self, X):
        self.X = X
        self.stats = self._mr()
    
    def _mr(self):
        return [self._mr_j(xj) for xj in T(self.X)]
    
    def _mr_j(self, xj):
        r = max(xj) - min(xj)
        m = mean(xj)
        return m, r

    def normalize(self):
        u = []
        for j, xj in enumerate(T(self.X)):
            m, r = self.stats[j]
            u.append([(xi - m) / r for xi in xj])
        return T(u)
        
    def denormalize(self, h_theta):
        h_thetad = []
        h_thetad.append(h_theta[0] - dot(h_theta[1:], [e[0] / e[1] for e in self.stats]))
        for elem, s in zip(h_theta[1:], self.stats):
            h_thetad.append(elem / s[1])
        return h_thetad


def prepend_x0(X):
    return [[1.] + e for e in X]


def cyclewshuffle(iterable):
    # cycle('ABCD') --> A B C D A B C D A B C D ...
    saved = []
    for element in iterable:
        yield element
        saved.append(element)
    while saved:
        random.shuffle(saved)
        for element in saved:
              yield element