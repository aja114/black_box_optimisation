Benchmarking of black-box optimisation techniques:
- Evolutionary Strategies
- Novelty Search - ES
- Quality diversity - ES
- Map-elite
- Covariance Matrix Adaptive - ES

The three functions tested are [Rastrigin](https://en.wikipedia.org/wiki/Rastrigin_function), [Ackley](https://en.wikipedia.org/wiki/Ackley_function) and [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function) which are pictured in 3D below

Ackley function

![ackley]( https://github.com/aja114/black_box_optimisation/blob/master/imgs/ackley.png "2D input ackley function")

Rastrigin function

![rastrigin]( https://github.com/aja114/black_box_optimisation/blob/master/imgs/rastrigin.png "2D input rastrigin function")

Rosenbrock function

![rosenbrock]( https://github.com/aja114/black_box_optimisation/blob/master/imgs/rosen.png "2D input rosenbrock function")


### Visual results

comparison between ES and QD algorithms on the ackley contour map

QD on ackley

![QD]( https://github.com/aja114/black_box_optimisation/blob/master/gif/ackley_qd.gif "QD")

ES on ackley

![ES]( https://github.com/aja114/black_box_optimisation/blob/master/gif/ackley_es.gif "ES")

ES on rosenbrock

![ES]( https://github.com/aja114/black_box_optimisation/blob/master/gif/rosen_es.gif "ES")


### Quantitative results

Results on Rosenbrock
![Rosenbrock]( https://github.com/aja114/black_box_optimisation/blob/master/imgs/rosen_line_plot.png "Rosenbrock Results")

Results on Ackley
![Ackley]( https://github.com/aja114/black_box_optimisation/blob/master/imgs/ackley_line_plot.png "Ackley Results")

Results on Rastrigin
![Rastrigin]( https://github.com/aja114/black_box_optimisation/blob/master/imgs/rastrigin_box_plot.png "Rastrigin Results")

Complete set of figures can be found by running the [script](https://github.com/aja114/black_box_optimisation/blob/master/results_plot.py)
