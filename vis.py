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

f = get_function(function_name)

f = Function(X_MIN, X_MAX, X_SHAPE, f)
f.make_data()
f.find_min()

plotter = Plotter(f.x, f.y)

fig_surf = plotter.surface()
fig_surf.savefig(f'imgs/{function_name}_3d_plot.png')
fig_cont, sc, sc2 = plotter.countour()

algo = get_algorithm(algorithm_name)
algo = algo(f)

anim = plotter.countour_animation(fig_cont, plotter.contour_update, algo)
plt.show()

# Save the animation
# anim.save(f'gif/{function_name}_{algorithm_name}.gif', writer='imagemagick', fps=1)
