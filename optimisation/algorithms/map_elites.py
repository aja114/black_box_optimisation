import numpy as np


def me():
    pass


def me_update(pos, x_range, x_min, x_max, x_shape,
              f, npop=25, sigma=0.4, alpha=0.1):
    r = int(npop**0.5)
    niche_range = x_range / r
    if len(pos['population']) == 0:
        for i in range(r):
            for j in range(r):
                s_i = i * niche_range + x_min
                e_i = s_i + niche_range
                s_j = j * niche_range + x_min
                e_j = s_j + niche_range
                init = np.concatenate(
                    (random_guess(niche_range, s_i, e_i, 1), random_guess(niche_range, s_j, e_j, 1)))
                pos['w_cand'].append(init)

    random_elite = pos['w_cand'][random.choice(range(npop))]
    mutated_elite = np.clip(random_elite + sigma *
                            np.random.randn(x_shape), x_min, x_max)

    cell = int(r * ((mutated_elite[0] - x_min) // niche_range) +
               (mutated_elite[1] - x_min) // niche_range)

    if f(mutated_elite) < f(pos['w_cand'][cell]):
        pos['w_cand'][cell] = mutated_elite
        pos['w'] = mutated_elite
