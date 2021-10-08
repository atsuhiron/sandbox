from typing import Tuple
from typing import Callable
from enum import Enum

import numpy as np


class ArgStatus(Enum):
    FREE = 0
    FREEZE = 1
    DEPEND = 2


class Argument:
    def __init__(self, name: str, parent_func_name: str,
                 value: float = 1.0, free_status: ArgStatus = None, bounds: Tuple[float, float] = None,
                 depend_func_name: str = None, depend_arg_name: str = None):
        self.name = name
        self.parent_func_name = parent_func_name
        self.value = value
        self.depend_func = None
        self.depend_info: Tuple[str, str] = (depend_func_name, depend_arg_name)

        if free_status is None:
            self.status = ArgStatus.FREE
        else:
            self.status = free_status

        if bounds is None:
            self.bounds = np.array([-np.inf, np.inf])
        else:
            self.bounds = np.array(bounds)

    def set_value(self, val: float):
        self.value = val

    def get_value(self) -> float:
        return self.value

    def set_depend_function(self, func: Callable[[float], float]):
        self.depend_func = func

    def get_depend_value(self, val: float) -> float:
        return self.depend_func(val)

    def get_status(self) -> ArgStatus:
        return self.status

    def set_status(self, status: ArgStatus):
        self.status = status

    def set_bound(self, lim_min: float = None, lim_max: float = None):
        if (lim_min is not None) and (lim_max is not None):
            if lim_min >= lim_max:
                print("Invalid bound: lim_min({0}) is larger than lim_max({1})".format(lim_min, lim_max))
                return

        if lim_min is not None:
            self.bounds[0] = lim_min
        if lim_max is not None:
            self.bounds[1] = lim_max

    def get_bounds(self) -> np.ndarray:
        return self.bounds