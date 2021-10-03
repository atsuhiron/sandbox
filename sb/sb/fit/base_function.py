from abc import *
from typing import List
from typing import Dict

import argument


class BaseFunction(metaclass=ABCMeta):
    def __init__(self):
        self.index = -1
        self.is_initialized = False

    @property
    @abstractmethod
    def func_name(self) -> str:
        pass

    @property
    @abstractmethod
    def argument_names(self) -> List[str]:
        pass

    @abstractmethod
    def f(self, *args):
        pass

    def get_index(self) -> int:
        return self.index

    def set_index(self, val: int):
        self.is_initialized = True
        self.index = val

    def get_unique_name(self) -> str:
        if self.is_initialized:
            return self.func_name + "_" + str(self.index)
        raise AssertionError("Index is not set.")

    def gen_arg_dict(self) -> Dict[str, argument.Argument]:
        return {name: argument.Argument(name, self.get_unique_name()) for name in self.argument_names}
