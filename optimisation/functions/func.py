import numpy as np

DATA_SIZE = 1000

class Function:
    def __init__(self, x_min, x_max, x_shape, f):
        self.f = f
        self.x_min = x_min
        self.x_max = x_max
        self.x_range = self.x_max - self.x_min
        self.x_shape = x_shape

    def make_data(self):
        x1, x2 = np.meshgrid(np.linspace(self.x_min, self.x_max, DATA_SIZE),
                             np.linspace(self.x_min, self.x_max, DATA_SIZE))
        self.x = np.stack([x1, x2], axis=-1)
        self.y = self.f(self.x)
        return self.x, self.y

    def find_min(self):
        i, j = (np.unravel_index(np.argmin(self.y), self.y.shape))
        self.y_sol = np.min(self.y)
        self.x_sol = (self.x[0, i, 0], self.x[j, 0, 1])
        print(
            f'Global minimum equal to {self.y_sol} found at ({self.x_sol[0]}, {self.x_sol[1]})')
        return self.x_sol, self.y_sol

    def random_guess(self):
        guess = np.random.rand(self.x_shape) * self.x_range + self.x_min
        return guess

    def __call__(self, x):
        return -self.f(x)
