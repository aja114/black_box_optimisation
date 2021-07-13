import numpy as np


def rosen(x):
    x0 = x[:-1]
    x1 = x[1:]
    y = np.sum(100*((x1 - x0**2)**2) + (x0-1)**2, axis=0)
    return y


def rastrigin(x):
    n = len(x)
    A = 10
    y = A*n + np.sum(x**2-A*np.cos(2*np.pi*x), axis=0)
    return y


def ackley(x):
    A = 20
    x1 = x[0]
    x2 = x[1]
    y = -A*np.exp(-0.2*np.sqrt(0.5*(x1**2+x2**2)))-np.exp(0.5 *
                                                          (np.cos(2*np.pi*x1)+np.cos(2*np.pi*x2)))+np.exp(1)+A
    return y

def sqr(x):
    y = np.sum(x**2, axis=0)
    return y

functions = {
    'rosen': rosen,
    'rastrigin': rastrigin,
    'ackley': ackley,
    'square': sqr
}
