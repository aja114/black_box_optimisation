import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm, ticker


class Plotter:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def surface(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(self.x[..., 0], self.x[..., 1], self.y,
                               cmap=cm.gist_heat_r, linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(ticker.LinearLocator(10))
        fig.colorbar(surf, shrink=0.5, aspect=5)

        return fig

    def countour(self):
        fig, ax = plt.subplots(1, 1, figsize=(13, 7))
        cp = ax.contourf(self.x[..., 0], self.x[..., 1], self.y,
                         locator=ticker.LogLocator(subs=range(1, 10)), cmap=cm.gist_heat_r)
        self.sc2 = ax.scatter([], [], s=2, c='b')
        self.sc = ax.scatter([], [], s=4, c='w')
        fig.colorbar(cp)

        return fig, self.sc, self.sc2

    def countour_animation(self, fig, update, *args):
        line_ani = animation.FuncAnimation(
            fig, update, 100, fargs=args, interval=50, blit=True)
        return line_ani

    def contour_update(self, num_iteration, alg):
        alg.one_step()
        self.sc.set_offsets(alg.x)
        self.sc2.set_offsets(alg.population)
        return self.sc2, self.sc
