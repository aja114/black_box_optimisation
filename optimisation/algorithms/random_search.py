from .alg import Algorithm


class RS(Algorithm):
    def __init__(self, function):
        super().__init__(function)
        self.x = self.f.random_guess()
        self.population = []

    def one_step(self):
        self.population.append(self.x)
        self.x = self.f.random_guess()
