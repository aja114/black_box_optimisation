import numpy as np
from .alg import Algorithm


class ES(Algorithm):
    def __init__(self, function, pop_size=200, sigma=0.1, alpha=0.005):
        super().__init__(function)
        self.pop_size = pop_size
        self.sigma = sigma
        self.alpha = alpha
        self.x = self.f.random_guess()
        self.population = []

    def one_step(self):
        noise = np.random.randn(self.pop_size, self.f.x_shape)
        self.population = self.f.clip(self.x + self.sigma * noise)
        fit = self.opposite_f(self.population)

        rewards = (fit - np.mean(fit)) / np.std(fit)
        offset = self.alpha / (self.pop_size * self.sigma) * \
            np.dot(noise.T, rewards)

        self.x = self.f.clip(self.x + offset)
