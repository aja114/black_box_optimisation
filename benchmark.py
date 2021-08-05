import sys
import numpy as np
import pandas as pd

from optimisation.visualisation import Plotter
from optimisation.utils import input_parse
from optimisation.algorithms import algorithms
from optimisation.functions import Function, functions

X_SHAPE = 2
X_MIN = -4
X_MAX = 4
NUM_EXP = 10
NUM_ITER = 1000

results = pd.DataFrame()

for f_name, f in functions.items():
    f = Function(X_MIN, X_MAX, X_SHAPE, f)
    f.make_data()
    f.find_min()

    for algo_name, algo in algorithms.items():
        print(f_name + '_' + algo_name)

        best_sols = np.zeros((NUM_EXP,))

        for i in range(NUM_EXP):
            algorithm = algo(f)
            best = algorithm.eval()
            for _ in range(NUM_ITER):
                algorithm.one_step()

                if algorithm.eval() < best:
                    best = algorithm.eval()

                if abs(best - f.y_sol) < 0.00001:
                    break

            best_sols[i] = best

        results[f_name + '_' + algo_name] = best_sols

results.to_csv('data/results.csv')
