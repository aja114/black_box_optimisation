import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from optimisation.utils.utils import input_parse, get_function, get_algorithm
from optimisation.visualisation import Plotter
from optimisation.functions import Function

X_SHAPE = 2
X_MIN = -5
X_MAX = 5

function_name, algorithm_name = input_parse(sys.argv)

for algorithm_name in ['rs', 'es', 'nses', 'qdes', 'me', 'cmaes']:
    f, algo = get_function(function_name), get_algorithm(algorithm_name)

    f = Function(X_MIN, X_MAX, X_SHAPE, f)
    f.make_data()
    f.find_min()

    algo = algo(f)

    plotter = Plotter(f.x, f.y)

    # fig_surf = plotter.surface()
    fig_cont, sc, sc2 = plotter.countour()

    def update(frame_num, alg):
        alg.one_step()
        sc.set_offsets(alg.x)
        sc2.set_offsets(alg.population)
        return sc2, sc

    pos = {'x': f.random_guess(), 'population': []}
    anim = plotter.animation(fig_cont, update, algo)
    plt.show()

# Save the animation
# anim.save(f'gif/{function}_{algorithm}.gif', writer='imagemagick', fps=5)
