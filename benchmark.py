import sys
import numpy as np
import pandas as pd

from optimisation.visualisation import plot_3d, plot_countour, plot_animation
from utils import input_parse
from opt_algos import random_guess, algos
from opt_functions import functions

x_shape = 2
x_min = -4
x_max = 4
x_range = x_max - x_min

NUM_EXP = 100
NUM_ITER = 1000

results = pd.DataFrame()

for f_name, f in functions.items():
    for algo_name, algo in algos.items():
        print(f_name + '_' + algo_name)

        x, y = make_data(f, x_min, x_max)

        x_sol, y_sol = find_min(x, y)

        print(
            f'Global minimum equal to {y_sol} found at ({x_sol[0]}, {x_sol[1]})')

        best_sols = np.zeros((NUM_EXP,))
        for i in range(NUM_EXP):
            pos = {'w': random_guess(
                x_range, x_min, x_max, x_shape), 'w_cand': []}
            best = f(pos['w'])
            for _ in range(NUM_ITER):
                algo(pos, x_range, x_min, x_max, x_shape, f)

                if f(pos['w']) < best:
                    best = f(pos['w'])

                if abs(best - y_sol) < 0.001:
                    break

            best_sols[i] = best

        results[f_name + '_' + algo_name] = best_sols

results.to_csv('results.csv')
