from opt_algos import algos
from opt_functions import functions


def input_parse(args):
    if len(args) > 2 and args[1] in functions.keys() and args[2] in algos.keys():
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
    algo = algos[algorithm]
    return algo
