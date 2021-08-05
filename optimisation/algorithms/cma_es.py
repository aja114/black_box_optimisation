import numpy as np


def cma_es_update(pos, function, npop=64):
    # Initialisation
    if len(pos) <= 3:
        # Initialisation
        mu = npop // 4

        C = np.eye(function.x_shape, function.x_shape).astype(np.float32)

        sigma = 0.3
        w = np.log(mu + 1 / 2) - \
            np.log(np.asarray(range(1, mu + 1))).astype(np.float32)
        w = w / np.sum(w)

        mu_eff = 1 / np.sum(w**2)

        pc = np.zeros(x_shape)
        ps = np.zeros(x_shape)

        cc = (4 + mu_eff / x_shape) / (4 + x_shape + 2 * mu_eff / x_shape)
        cs = (mu_eff + 2) / (x_shape + mu_eff + 5)
        c1 = 2 / ((x_shape + 1.3) ** 2 + mu_eff)
        cmu = np.min(
            [1 - c1, 2 * ((mu_eff - 2 + (1 / mu_eff)) / ((x_shape + 2)**2 + mu_eff))])
        ds = 1 + 2 * \
            np.max([0, np.sqrt((mu_eff - 1) / (x_shape + 1)) - 1]) + cs
        exp_length_gauss = np.sqrt(
            x_shape) * (1 - 1 / (4 * x_shape) + 1 / (21 * x_shape**2))
        C = np.triu(C) + np.triu(C, 1).T
        D, B = np.linalg.eig(C)
        D = np.sqrt(D)
        C_sqrtinv = B.dot(np.diag(D ** -1).dot(B.T))

        pos["mu"] = mu
        pos["C"] = C
        pos["sigma"] = sigma
        pos["weights"] = w
        pos["mu_eff"] = mu_eff
        pos["pc"] = pc
        pos["ps"] = ps
        pos["cc"] = cc
        pos["cs"] = cs
        pos["c1"] = c1
        pos["cmu"] = cmu
        pos["ds"] = ds
        pos["exp_length_gauss"] = exp_length_gauss
        pos["C_sqrtinv"] = C_sqrtinv

    mu = pos["mu"]
    C = pos["C"]
    sigma = pos["sigma"]
    w = pos["weights"]
    mu_eff = pos["mu_eff"]
    pc = pos["pc"]
    ps = pos["ps"]
    cc = pos["cc"]
    cs = pos["cs"]
    c1 = pos["c1"]
    cmu = pos["cmu"]
    ds = pos["ds"]
    exp_length_gauss = pos["exp_length_gauss"]
    C_sqrtinv = pos["C_sqrtinv"]
    m = pos["w"]

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

    y = np.random.multivariate_normal(np.zeros(x_shape), C, size=npop)
    samples = m + sigma * y
    samples = np.clip(samples, x_min, x_max)
    pos["w_cand"] = samples

    fit = f(samples.T)

    order = np.argsort(fit)[:mu]
    samples = samples[order, :]
    y = y[order, :]

    yw = w.dot(y)

    # Update the popuation mean
    m = np.clip(m + sigma * yw, x_min, x_max)

    # Update the parameters

    C = np.triu(C) + np.triu(C, 1).T
    D, B = np.linalg.eig(C)
    D = np.sqrt(D)
    pos["C_sqrtinv"] = B.dot(np.diag(D ** -1).dot(B.T))

    pos["ps"] = (1 - cs) * ps + np.sqrt(cs * (2 - cs)
                                        * mu_eff) * C_sqrtinv.dot(yw)

    # pos["hsig"] = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (i+1))) < ((1.4 + 2 / (x_shape + 1)) * exp_length_gauss)
    # pos["pc"] = (1 - cc) * pc + 1 * hsig * \
    #     np.sqrt(cc * (2 - cc) * mu_eff) * yw
    hsig = 0
    pos["pc"] = (1 - cc) * pc + 1 * (np.linalg.norm(ps) < (1.5 * npop**0.5)) * \
        np.sqrt(cc * (2 - cc) * mu_eff) * yw

    pos["C"] = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C) + \
        cmu * y.T.dot(np.diag(w)).dot(y)

    pos["sigma"] = sigma * \
        np.exp((cs / ds) * (np.linalg.norm(ps) / exp_length_gauss - 1))

    pos["w"] = m
