class Algorithm:
    def __init__(self, function):
        self.f = function
        self.x = self.f.random_guess()

    def one_step(self):
        pass

    def eval(self):
        return self.f(self.x)
