from typing import List
from collections import Counter
import abc
import dataclasses


class ILoad(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def get_load(self) -> float:
        pass

    def __eq__(self, other):
        if not isinstance(other, ILoad):
            return False
        return self.get_load() == other.get_load()

    def __lt__(self, other):
        if not isinstance(other, ILoad):
            raise NotImplementedError
        return self.get_load() < other.get_load()


@dataclasses.dataclass
class SimpleLoad(ILoad):
    load: float

    def get_load(self) -> float:
        return self.load


@dataclasses.dataclass
class MultipliedLoad(ILoad):
    load: float
    coef: float = 1.0

    def get_load(self) -> float:
        return self.load * self.coef


class Separator(ILoad):
    def get_load(self) -> float:
        return 0.0


def _to_hashable(counter: Counter) -> tuple:
    counter = dict(counter)
    elements = []
    for k, v in zip(counter.keys(), counter.values()):
        elements.append((k, v))
    return tuple(elements)


def are_equivalent_loads(loads1: List[List[ILoad]], loads2: List[List[ILoad]]) -> bool:
    if len(loads1) != len(loads2):
        return False

    counters1 = set([_to_hashable(Counter([load.get_load() for load in load_set])) for load_set in loads1])
    counters2 = set([_to_hashable(Counter([load.get_load() for load in load_set])) for load_set in loads2])
    return counters1 == counters2
