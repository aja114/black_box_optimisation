import numpy as np
from sklearn.neighbors import NearestNeighbors


def nses(x, archive, function, K, npop=100, sigma=0.5, alpha=0.15):
    noise = np.random.randn(npop, function.x_shape)
    population = np.clip(x + sigma * noise, function.x_min, function.x_max)
    neighbors = NearestNeighbors(
        n_neighbors=K,
        algorithm='ball_tree').fit(archive)
    fit = np.mean(neighbors.kneighbors(population)[0], axis=1)
    rewards = (fit - np.mean(fit)) / np.std(fit)
    offset = alpha / (npop * sigma) * np.dot(noise.T, rewards)

    x = np.clip(x + offset, function.x_min, function.x_max)

    archive.append(x)
    return x, archive


def nses_update(pos, function, npop=50, sigma=0.5, alpha=0.15):
    K = 5
    if len(pos['population']) < K:
        pos['population'] += [function.random_guess() for _ in range(K)]

    x = pos['x']
    archive = pos['population']

    x, archive = nses(x, archive, function, K)

    pos['x'] = x
    pos['population'] = archive
