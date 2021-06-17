import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from create_data import make_data, find_min
from plot_func import plot_3d, plot_countour, plot_animation
from utils import input_parse, get_function, get_algorithm
from opt_algos import random_guess


def update(frame_num, algo, pos, x_range, x_min, x_max, x_shape, f):
    w_cand = algo(pos, x_range, x_min, x_max, x_shape, f)
    sc.set_offsets(pos['w'])
    sc2.set_offsets(pos['w_cand'])
    return sc2, sc


x_shape = 2
x_min = -5
x_max = 5
x_range = x_max - x_min


function, algorithm = input_parse(sys.argv)
f, algo = get_function(function), get_algorithm(algorithm)

x, y = make_data(f, x_min, x_max)

x_sol, y_sol = find_min(x, y)

print(f'Global minimum equal to {y_sol} found at ({x_sol[0]}, {x_sol[1]})')

# fig_3d = plot_3d(x, y)
fig_cont, sc, sc2 = plot_countour(x, y)

pos = {'w': random_guess(x_range, x_min, x_max, x_shape), 'w_cand': []}
anim = plot_animation(fig_cont, update, algo, pos, x_range,
                      x_min, x_max, x_shape, f)

# plt.show()

# Save the animation
anim.save(f'gif/{function}_{algorithm}.gif', writer='imagemagick', fps=5)
