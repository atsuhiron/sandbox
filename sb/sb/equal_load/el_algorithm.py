from typing import List
from typing import Tuple
import itertools
import abc
import multiprocessing.pool as mp

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
        #with mp.ThreadPool(20) as pool:
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