from .evolutionary_strategies import ES
from .novelty_search import NSES
from .quality_diversity import QDES
from .random_search import RS
from .map_elites import ME
from .cma_es import CMAES

algorithms = {
    'rs': RS,
    'es': ES,
    'nses': NSES,
    'qdes': QDES,
    'me': ME,
    'cmaes': CMAES
}
