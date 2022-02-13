from typing import List

import numpy as np
import matplotlib.pyplot as plt

from sb.equal_load.load import SimpleLoad
from sb.equal_load.load import are_equivalent_loads
from sb.equal_load.el_algorithm import *
import sb.equal_load.graphix as gfx


def gen_data(num: int, index: float = None) -> List[SimpleLoad]:
    random_load = (np.random.random(num) + 1) * (10 ** np.random.randint(1, 3, num))
    if index is None:
        return [SimpleLoad(load) for load in random_load]
    random_load = np.power(random_load, index)
    return [SimpleLoad(load) for load in random_load]


if __name__ == "__main__":
    __spec__ = None
    c_num = 3
    for _ in range(1):
        ll = gen_data(10, 1)
        alg_full = ELAFullSearch(c_num, ll)
        alg_greedy = ELAGreedySearch(c_num, ll)
        score_f, best_ll_f = alg_full.search()
        score_g, best_ll_g = alg_greedy.search()
        gfx.show_load(best_ll_f)
        gfx.show_load(best_ll_g)
        # if not are_equivalent_loads(best_ll_f, best_ll_g):
        #     show_2load(best_ll_f, best_ll_g)

