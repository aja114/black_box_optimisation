import numpy as np

DATA_SIZE = 1000


def make_data(f, x_min, x_max):
    X, Y = np.meshgrid(np.linspace(x_min, x_max, DATA_SIZE),
                       np.linspace(x_min, x_max, DATA_SIZE))
    x = np.stack([X, Y], axis=0)
    y = f(x)

    return x, y


def find_min(x, y):
    i, j = (np.unravel_index(np.argmin(y), y.shape))
    y_min = np.min(y)
    x_min = (x[0, 0, i], x[1, j, 0])
    return x_min, y_min
