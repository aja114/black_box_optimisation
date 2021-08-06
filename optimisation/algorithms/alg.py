class Algorithm:
    def __init__(self, function):
        self.f = function
        self.x = self.f.random_guess()
        self.population = []

    def one_step(self):
        raise NotImplementedError("Subclasses should implement this!")

    def search_loop(self, num_iterations, tol=1e-5):
        self.score = self.eval()

        for _ in range(num_iterations):
            self.one_step()
            self.score = min(self.score, self.eval())

            if abs(self.score - self.f.y_sol) < tol:
                break

        # print(f"estimate: {self.x}, fitness: {self.eval()}")
        return self.score

    def eval(self):
        return self.f(self.x)

    def opposite_f(self, x):
        return -self.f(x)
