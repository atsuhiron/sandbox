from abc import *


class BaseFunction(metaclass=ABCMeta):
    @abstractmethod
    def f(self, *args):
        pass

    @abstractmethod
    def d_f(self, *args):
        pass