import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

def random_guess(x_range, x_min, x_max, x_shape):
    return (np.random.rand(x_shape) * x_range) + x_min


def es_update(pos, x_range, x_min, x_max, x_shape, f, npop=200, sigma=0.1, alpha=0.005):
    w = pos['w']
    N = np.random.randn(npop, x_shape)
    R = np.zeros(npop)
    w_try = w + sigma * N
    R = -f(w_try.T)
    A = (R - np.mean(R)) / np.std(R)
    offset = alpha/(npop*sigma) * np.dot(N.T, A)
    pos['w'] = w + offset
    pos['w_cand'] = w_try


def rs_update(pos, x_range, x_min, x_max, x_shape, f):
    w = pos['w']
    pos['w_cand'].append(pos['w'])
    pos['w'] = random_guess(x_range, x_min, x_max, x_shape)


def ns_update(pos, x_range, x_min, x_max, x_shape, f, npop=50, sigma=0.5, alpha=0.15):
    K = 7

    w = pos['w']
    pos['w_cand'].append(pos['w'])
    if len(pos['w_cand']) < K:
        pos['w_cand'] += [random_guess(x_range, x_min, x_max, x_shape) for _ in range(K)]

    N = np.random.randn(npop, x_shape)
    R = np.zeros(npop)
    w_try = np.clip(w + sigma * N, x_min, x_max)
    archive = np.array(pos['w_cand'])

    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(archive)
    R = np.mean(nbrs.kneighbors(w_try)[0], axis=1)

    A = (R - np.mean(R)) / np.std(R)
    offset = alpha/(npop*sigma) * np.dot(N.T, A)

    pos['w'] = np.clip(w + offset, x_min, x_max)


def qd_update(pos, x_range, x_min, x_max, x_shape, f, npop=50, sigma=0.5, alpha=0.1):
    K = 7

    w = pos['w']
    pos['w_cand'].append(pos['w'])
    if len(pos['w_cand']) < K:
        pos['w_cand'] += [random_guess(x_range, x_min, x_max, x_shape) for _ in range(K)]

    N = np.random.randn(npop, x_shape)
    R1 = np.zeros(npop)
    R2 = np.zeros(npop)

    w_try = np.clip(w + sigma * N, x_min, x_max)

    R1 = -f(w_try.T)

    archive = np.array(pos['w_cand'])
    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(archive)
    R2 = np.mean(nbrs.kneighbors(w_try)[0], axis=1)

    R = normalize_array(R1) + normalize_array(R2)

    A = (R - np.mean(R)) / np.std(R)
    offset = alpha/(npop*sigma) * np.dot(N.T, A)

    pos['w'] = np.clip(w + offset, x_min, x_max)

def me_update(pos, x_range, x_min, x_max, x_shape, f, npop=25, sigma=0.4, alpha=0.1):
    r = int(npop**0.5)
    niche_range = x_range / r
    if len(pos['w_cand']) == 0:
        for i in range(r):
            for j in range(r):
                s_i = i * niche_range + x_min
                e_i = s_i + niche_range
                s_j = j * niche_range + x_min
                e_j = s_j + niche_range
                init = np.concatenate((random_guess(niche_range, s_i, e_i, 1), random_guess(niche_range, s_j, e_j, 1)))
                pos['w_cand'].append(init)

    random_elite = pos['w_cand'][random.choice(range(npop))]
    mutated_elite = np.clip(random_elite + sigma * np.random.randn(x_shape), x_min, x_max)

    cell = int(r*((mutated_elite[0]-x_min)//niche_range) + (mutated_elite[1]-x_min)//niche_range)

    if f(mutated_elite) < f(pos['w_cand'][cell]):
        pos['w_cand'][cell] = mutated_elite
        pos['w'] = mutated_elite


def normalize_array(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

algos = {
    'rs': rs_update,
    'es': es_update,
    'ns': ns_update,
    'qd': qd_update,
    'me': me_update
}
