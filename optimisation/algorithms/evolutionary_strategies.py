import numpy as np


def es(x, function, npop=200, sigma=0.1, alpha=0.005):
    noise = np.random.randn(npop, function.x_shape)
    population = np.clip(x + sigma * noise, function.x_min, function.x_max)
    fit = function(population)

    rewards = (fit - np.mean(fit)) / np.std(fit)
    offset = alpha / (npop * sigma) * np.dot(noise.T, rewards)

    x_upd = x + offset

    return x_upd, population


def es_update(pos, function, npop=200, sigma=0.1, alpha=0.005):
    x = pos['x']
    x, population = es(x, function, npop, sigma, alpha)
    pos['x'] = x
    pos['population'] = population
