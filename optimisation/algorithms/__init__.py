from .evolutionary_strategies import es_update
from .novelty_search import nses_update
from .quality_diversity import qdes_update
from .random_search import rs_update
from .map_elites import me_update
from .cma_es import cma_es_update

algorithms = {
    'rs': rs_update,
    'es': es_update,
    'nses': nses_update,
    'qdes': qdes_update,
    'me': me_update,
    'cmaes': cma_es_update
}
