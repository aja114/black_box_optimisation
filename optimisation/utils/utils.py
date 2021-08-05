from ..algorithms import algorithms
from ..functions import functions

def input_parse(args):
    if len(args) > 2 and args[1] in functions.keys(
    ) and args[2] in algorithms.keys():
        function = args[1]
        algorithm = args[2]
    else:
        function = 'rosen'
        algorithm = 'es'

    return function, algorithm


def get_function(function):
    print(f'function {function}')
    f = functions[function]
    return f


def get_algorithm(algorithm):
    print(f'algorithm {algorithm}')
    algo = algorithms[algorithm]
    return algo
