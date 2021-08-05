import numpy as np
import random

from .alg import Algorithm


class ME(Algorithm):
    def __init__(self, function, pop_size=25, sigma=0.5):
        super().__init__(function)
        self.pop_size = int(pop_size**0.5)**2
        self.sigma = sigma
        self.x = self.f.random_guess()
        self.population = []

        # Populate the niches
        self.init_grid()

    def init_grid(self):
        self.grid_size = int(self.pop_size**0.5)
        self.niche_range = self.f.x_range / self.grid_size

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                s_i = i * self.niche_range + self.f.x_min
                e_i = s_i + self.niche_range
                s_j = j * self.niche_range + self.f.x_min
                e_j = s_j + self.niche_range

                centers = np.array([[s_j, s_i], [e_j, e_i]])
                niche = np.mean(centers, axis=0)

                self.population.append(niche)

    def one_step(self):
        random_elite = random.choice(self.population)
        noise = np.random.randn(self.f.x_shape)
        mutated_elite = self.f.clip(random_elite + self.sigma * noise)
        self.x = mutated_elite

        cell = self.get_cell(mutated_elite)

        if self.f(mutated_elite) > self.f(self.population[cell]):
            self.population[cell] = mutated_elite

    def get_cell(self, x):
        cell_x = (x[0] * 0.999 - self.f.x_min) // self.niche_range
        cell_y = (x[1] * 0.999 - self.f.x_min) // self.niche_range
        cell = int(self.grid_size * cell_y + cell_x)

        return cell
