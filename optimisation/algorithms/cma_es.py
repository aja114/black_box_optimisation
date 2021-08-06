import numpy as np

from .alg import Algorithm


class CMAES(Algorithm):
    def __init__(self, function, pop_size=50, sigma=0.3):
        super().__init__(function)
        self.iteration = 0

        self.pop_size = pop_size
        self.mu = self.pop_size // 4

        self.C = np.eye(self.f.x_shape, self.f.x_shape).astype(np.float32)

        self.sigma = sigma

        w = np.log(self.mu + 1 / 2) - \
            np.log(np.asarray(range(1, self.mu + 1))).astype(np.float32)
        self.w = w / np.sum(w)

        self.mu_eff = 1 / np.sum(self.w**2)
        self.hsig = 0
        self.pc = np.zeros(self.f.x_shape)
        self.ps = np.zeros(self.f.x_shape)

        self.cc = (4 + self.mu_eff / self.f.x_shape) / \
            (4 + self.f.x_shape + 2 * self.mu_eff / self.f.x_shape)
        self.cs = (self.mu_eff + 2) / (self.f.x_shape + self.mu_eff + 5)
        self.c1 = 2 / ((self.f.x_shape + 1.3) ** 2 + self.mu_eff)
        self.cmu = np.min(
            [1 - self.c1, 2 * ((self.mu_eff - 2 + (1 / self.mu_eff)) / ((self.f.x_shape + 2)**2 + self.mu_eff))])
        self.ds = 1 + 2 * \
            np.max([0, np.sqrt((self.mu_eff - 1) /
                               (self.f.x_shape + 1)) - 1]) + self.cs
        self.exp_length_gauss = np.sqrt(
            self.f.x_shape) * (1 - 1 / (4 * self.f.x_shape) + 1 / (21 * self.f.x_shape**2))

        self.x = self.f.random_guess()
        self.population = []

    def one_step(self):
        self.iteration += 1
        # print("m: ", m, "shape: ", m.shape)
        # print("p_cov: ", pc, "shape: ", pc.shape)
        # print("hsig: ", hsig)
        # print("p_sigma: ", ps, "shape: ", ps.shape)
        # print("Cov: ", C, "shape: ", C.shape)
        # print("C_sqrtinv: ", C_sqrtinv)
        # print("sigma: ", sigma)
        # print("fit: ", fit.shape)#, "shape: ", fit.shape)
        # print("order: ", order.shape)#, "shape: ", order.shape)
        # print("samples: ", samples.shape)#, "shape: ", samples.shape)
        # print("y: ", y.shape)#, "shape: ", y.shape)
        # print("yw: ", yw, "shape: ", yw.shape)
        y = np.random.multivariate_normal(
            np.zeros(self.f.x_shape), self.C, size=self.pop_size)
        self.population = self.f.clip(self.x + self.sigma * y)

        fit = self.opposite_f(self.population)

        order = np.argsort(fit)[-self.mu:][::-1]
        y = y[order, :]

        yw = self.w.dot(y)

        # Update the popuation mean
        x = self.f.clip(self.x + self.sigma * yw)

        # Update the parameters
        C = np.triu(self.C) + np.triu(self.C, 1).T
        D, B = np.linalg.eig(C)
        D = np.sqrt(D)
        self.C_sqrtinv = B.dot(np.diag(D ** -1).dot(B.T))

        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs)
                                                    * self.mu_eff) * self.C_sqrtinv.dot(yw)

        self.hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * (
            self.iteration + 1))) < ((1.4 + 2 / (self.f.x_shape + 1)) * self.exp_length_gauss)
        self.pc = (1 - self.cc) * self.pc + 1 * self.hsig * \
            np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * yw

        # self.pc = (1 - self.cc) * self.pc + 1 * (np.linalg.norm(self.ps) < (1.5 * self.pop_size**0.5)) * \
        #     np.sqrt(self.cc * (2 - self.cc) * self.mu_eff) * yw

        self.C = (1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - self.hsig) * self.cc * (2 - self.cc) * self.C) + \
            self.cmu * y.T.dot(np.diag(self.w)).dot(y)

        self.sigma = self.sigma * \
            np.exp((self.cs / self.ds) *
                   (np.linalg.norm(self.ps) / self.exp_length_gauss - 1))

        self.x = x
