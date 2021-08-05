import numpy as np
from sklearn.neighbors import NearestNeighbors

from .alg import Algorithm


class NSES(Algorithm):
    def __init__(self, function, K=7, pop_size=200, sigma=0.1, alpha=0.01):
        super().__init__(function)
        self.pop_size = pop_size
        self.sigma = sigma
        self.alpha = alpha
        self.K = K
        self.x = self.f.random_guess()
        self.population = [self.f.random_guess() for _ in range(self.K)]

    def one_step(self):
        noise = np.random.randn(self.pop_size, self.f.x_shape)
        curr_pop = self.f.clip(self.x + self.sigma * noise)
        neighbors = NearestNeighbors(
            n_neighbors=self.K,
            algorithm='ball_tree').fit(self.population)
        fit = np.mean(neighbors.kneighbors(curr_pop)[0], axis=1)
        rewards = (fit - np.mean(fit)) / np.std(fit)
        offset = self.alpha / (self.pop_size * self.sigma) * \
            np.dot(noise.T, rewards)

        self.x = self.f.clip(self.x + offset)

        self.population.append(self.x)
