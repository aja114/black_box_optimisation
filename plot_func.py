import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from matplotlib import cm, ticker


def plot_3d(x, y):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(x[0], x[1], y, cmap=cm.gist_heat_r,
                           linewidth=0, antialiased=False)
    ax.zaxis.set_major_locator(ticker.LinearLocator(10))
    fig.colorbar(surf, shrink=0.5, aspect=5)

    return fig


def plot_countour(x, y):
    fig, ax = plt.subplots(1, 1, figsize=(13, 7))
    cp = ax.contourf(x[0], x[1], y, locator=ticker.LogLocator(
        subs=range(1, 10)), cmap=cm.gist_heat_r)
    sc2 = ax.scatter([], [], s=2, c='b')
    sc = ax.scatter([], [], s=4, c='w')
    fig.colorbar(cp)

    return fig, sc, sc2


def plot_animation(fig, update, *args):
    line_ani = animation.FuncAnimation(
        fig, update, 100, fargs=args, interval=50, blit=True)
    return line_ani
