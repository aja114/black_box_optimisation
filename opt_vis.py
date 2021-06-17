import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from create_data import make_data, find_min
from plot_func import plot_3d, plot_countour, plot_animation
from utils import input_parse
from opt_algos import random_guess


def update(frame_num, algo, pos, x_range, x_min, x_max, x_shape, f):
    w_cand = algo(pos, x_range, x_min, x_max, x_shape, f)
    sc.set_offsets(pos['w'])
    sc2.set_offsets(pos['w_cand'])
    return sc, sc2


x_shape = 2
x_min = -5
x_max = 5
x_range = x_max - x_min

f, algo = input_parse(sys.argv)

x, y = make_data(f, x_min, x_max)

x_sol, y_sol = find_min(x, y)

print(f'Global minimum equal to {y_sol} found at ({x_sol[0]}, {x_sol[1]})')

# fig_3d = plot_3d(x, y)
fig_cont, sc, sc2 = plot_countour(x, y)

pos = {'w': random_guess(x_range, x_min, x_max, x_shape), 'w_cand': []}
anim = plot_animation(fig_cont, update, algo, pos, x_range,
                      x_min, x_max, x_shape, f)

plt.show()

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
anim.save('gif/im.mp4', writer=writer)
#anim.save('gif/anim.gif', writer='imagemagick', fps=5)
