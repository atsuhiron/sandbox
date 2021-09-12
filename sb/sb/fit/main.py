from typing import Tuple
from typing import Union
import matplotlib.pyplot as plt
import numpy as np

import base_function


class ExampleFunc(base_function.BaseFunction):
    def __init__(self):
        pass

    def f(self, *args):
        return args[1] * args[0] ** 4 + \
        args[2] * args[0] ** 3 + \
        args[3] * args[0] ** 2 + \
        args[4] * args[0] + \
        args[5]

    def d_f(self, *args):
        return 4 * args[1] * args[0] ** 3 + \
        3 * args[2] * args[0] ** 2 + \
        2 * args[3] * args[0] + \
        args[4]


