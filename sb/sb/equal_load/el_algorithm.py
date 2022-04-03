from typing import List
from typing import Tuple
import itertools
import abc
import multiprocessing.pool as mp
import random
import copy

from tqdm import tqdm
from scipy.special import perm
import numpy as np

from sb.equal_load.load import ILoad
from sb.equal_load.load import Separator


class ELABase(metaclass=abc.ABCMeta):
    def __init__(self, num: int, load_list: List[ILoad]):
        self.num = num
        self.load_list = load_list

    def get_result_when_less_pattern(self) -> Tuple[float, List[List[ILoad]]]:
        carrier = [[load] for load in self.load_list]
        score = ELABase.evaluate_load(carrier)
        if len(carrier) < self.num:
            pad = self.num - len(carrier)
            carrier.append([] * pad)
        return score, carrier

    @abc.abstractmethod
    def search(self) -> Tuple[float, List[List[ILoad]]]:
        pass

    @staticmethod
    def evaluate_load(loads: List[List[ILoad]]) -> float:
        return max([sum([a_load.get_load() for a_load in load_set]) for load_set in loads])


class ELAFullSearch(ELABase):
    def search(self) -> Tuple[float, List[List[ILoad]]]:
        if len(self.load_list) <= self.num:
            return self.get_result_when_less_pattern()

        sep = [Separator()] * (self.num - 1)
        permu = itertools.permutations(self.load_list + sep)
        permu_num = len(self.load_list) + self.num - 1
        total = int(perm(permu_num, permu_num, True))

        best_pattern = None
        best_score = float("inf")
        # with mp.ThreadPool(20) as pool:
        with mp.Pool(16) as pool:
            with tqdm(total=total) as pbar:
                for res in pool.imap_unordered(
                        ELAFullSearch.assign_and_eval,
                        ELAFullSearch.arg_gen(self.num, permu),
                        chunksize=500):
                    score, carrier = res
                    if score < best_score:
                        best_score = score
                        best_pattern = carrier
                    pbar.update(1)
        return best_score, best_pattern

    @staticmethod
    def arg_gen(num: int, permutations: itertools.permutations):
        for permutation in permutations:
            yield num, permutation

    @staticmethod
    def assign_load(num: int, load_list: Tuple[ILoad, ...]) -> List[List[ILoad]]:
        carrier = [[] for _ in range(num)]
        ci = 0
        for li in range(len(load_list)):
            load = load_list[li]
            if isinstance(load, Separator):
                ci += 1
                continue
            carrier[ci].append(load)
        return carrier

    @staticmethod
    def assign_and_eval(arg: Tuple[int, Tuple[ILoad, ...]]) -> Tuple[float, List[List[ILoad]]]:
        num, load_list = arg
        carrier = ELAFullSearch.assign_load(num, load_list)
        return ELAFullSearch.evaluate_load(carrier), carrier


class ELAGreedySearch(ELABase):
    @staticmethod
    def find_lightest_carrier_index(loads: List[List[ILoad]]) -> int:
        load_sum = np.array([sum([load.get_load() for load in load_set]) for load_set in loads])
        return int(np.argmin(load_sum))

    def search(self) -> Tuple[float, List[List[ILoad]]]:
        if len(self.load_list) <= self.num:
            return self.get_result_when_less_pattern()

        self.load_list = sorted(self.load_list, reverse=True)
        carrier = [[] for _ in range(self.num)]
        for load in tqdm(self.load_list):
            lightest_index = ELAGreedySearch.find_lightest_carrier_index(carrier)
            carrier[lightest_index].append(load)
        return ELAGreedySearch.evaluate_load(carrier), carrier


class ELARandomGreedySearch(ELABase):
    def search(self) -> Tuple[float, List[List[ILoad]]]:
        n = 3
        max_iter = self.num * len(self.load_list) * 10000
        result_list = []
        for _ in tqdm(range(n)):
            random_loads = self._random_init()
            moe_count = 0
            score = ELARandomGreedySearch.evaluate_load(random_loads)
            while moe_count < max_iter:
                #moe = random.random() > 0.5
                if True:
                    moe_loads = self._move_load(random_loads)
                else:
                    moe_loads = self._exchange_load(random_loads)

                moe_score = ELARandomGreedySearch.evaluate_load(moe_loads)
                if moe_score < score:
                    score = moe_score
                    moe_count = 0
                    random_loads = moe_loads
                else:
                    moe_count += 1
            result_list.append((score, random_loads))
        return min(result_list, key=lambda pair: pair[0])

    def _random_load_nums(self) -> List[int]:
        load_num = len(self.load_list)
        max_load_on_a_carrier = load_num - self.num + 1

        random_balanced_load_num = []
        for ii in range(self.num - 1):
            load_on_a_carrier = random.randint(1, max_load_on_a_carrier)
            random_balanced_load_num.append(load_on_a_carrier)
            max_load_on_a_carrier -= (load_on_a_carrier - 1)
        random_balanced_load_num.append(load_num - sum(random_balanced_load_num))
        return random_balanced_load_num

    def _random_init(self) -> List[List[ILoad]]:
        random_loads = copy.deepcopy(self.load_list)
        random.shuffle(random_loads)

        random_load_nums = self._random_load_nums()
        carrier = [[] for _ in range(self.num)]
        l_index = 0
        for c in range(self.num):
            for _ in range(random_load_nums[c]):
                carrier[c].append(random_loads[l_index])
                l_index += 1
        return carrier

    def _gen_from_to_index(self, carrier: List[List[ILoad]]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        from_c_index = -1
        to_c_index = -1
        from_c_has_load_more_than_1 = False
        while not ((from_c_index != to_c_index) and from_c_has_load_more_than_1):
            from_c_index = random.randint(0, self.num - 1)
            to_c_index = random.randint(0, self.num - 1)
            from_c_has_load_more_than_1 = len(carrier[from_c_index]) > 1

        from_l_index = random.randint(0, len(carrier[from_c_index]) - 1)
        to_l_index = random.randint(0, len(carrier[to_c_index]) - 1)
        return (from_c_index, from_l_index), (to_c_index, to_l_index)

    def _move_load(self, carrier: List[List[ILoad]]):
        froms, tos = self._gen_from_to_index(carrier)
        pop_load = carrier[froms[0]].pop(froms[1])
        carrier[tos[0]].append(pop_load)
        return carrier

    def _exchange_load(self, carrier: List[List[ILoad]]):
        froms, tos = self._gen_from_to_index(carrier)
        pop_load1 = carrier[froms[0]].pop(froms[1])
        pop_load2 = carrier[tos[0]].pop(tos[1])
        carrier[tos[0]].append(pop_load1)
        carrier[froms[0]].append(pop_load2)
        return carrier


if __name__ == "__main__":
    import pprint
    from sb.equal_load.load import SimpleLoad

    ll = [SimpleLoad(load=16.253557774074658),
          SimpleLoad(load=17.61833351439079),
          SimpleLoad(load=193.46259669182442),
          SimpleLoad(load=150.30559960567024),
          SimpleLoad(load=15.013089710173572),
          SimpleLoad(load=14.103840588683125),
          SimpleLoad(load=16.865148941365117),
          SimpleLoad(load=18.965240668296694),
          SimpleLoad(load=15.060557413770358),
          SimpleLoad(load=13.867308240998799)]
    ela_rgs = ELARandomGreedySearch(3, ll)
    ela_rgs.search()