import numpy as np


def rs(function):
    x = function.random_guess()
    return x


def rs_update(pos, function):
    x = rs(function)
    population = pos['popuation']
    population.append(x)
    pos['x'] = x
