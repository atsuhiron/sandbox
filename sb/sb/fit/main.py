from typing import Tuple
from typing import Dict
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as so

import base_function
import data_generator
import argument


class ExampleFunc(base_function.BaseFunction):
    @property
    def func_name(self) -> str:
        return "quartic"

    @property
    def argument_names(self) -> List[str]:
        return ["a", "b", "c", "d", "e"]

    def __init__(self):
        super().__init__()
        self.arguments = self.gen_arg_dict()

    def f(self, *args):
        return args[1] * args[0] ** 4 + \
               args[2] * args[0] ** 3 + \
               args[3] * args[0] ** 2 + \
               args[4] * args[0] + \
               args[5]


class Functions:
    def __init__(self, funcs: List[base_function.BaseFunction]):
        self.funcs = funcs
        self._set_function_index()
        self.arg_dict = self._gen_arg_dict(self.funcs)

    def f(self, *args):
        self._assign_args(*args)

        res_values = []
        for i, f_name in enumerate(self.arg_dict):
            funclet_args = [self.arg_dict[f_name][a_name].get_value() for a_name in self.arg_dict[f_name]]
            res_values.append(self.funcs[i].f(*funclet_args))
        return sum(res_values)

    def _assign_args(self, *args):
        arg_index = 0
        for f_name in self.arg_dict:
            for a_name in self.arg_dict[f_name]:
                target_arg = self.arg_dict[f_name][a_name]
                if target_arg.status == argument.ArgStatus.FREE:
                    target_arg.set_value(args[arg_index])
                    arg_index += 1

        for f_name in self.arg_dict:
            for a_name in self.arg_dict[f_name]:
                target_arg = self.arg_dict[f_name][a_name]
                if target_arg.status == argument.ArgStatus.DEPEND:
                    depend_arg = self.arg_dict[target_arg.depend_info[0]][target_arg.depend_info[1]]
                    val = target_arg.get_depend_value(depend_arg.get_value())
                    target_arg.set_value(val)

    def _set_function_index(self):
        last_index: Dict[str, int] = {}
        for func in self.funcs:
            name = func.func_name
            if last_index.get(name) is None:
                last_index[name] = 0
                func.set_index(0)
            else:
                last_index[name] += 1
                func.set_index(last_index[name])

    @staticmethod
    def _gen_arg_dict(funcs: List[base_function.BaseFunction]) -> Dict[str, Dict[str, argument.Argument]]:
        func_d = {}
        for f in funcs:
            func_d[f.get_unique_name()] = f.gen_arg_dict()
        return func_d


if __name__ == "__main__":
    import importlib
    importlib.reload(base_function)

    num = 128
    np.random.seed(213)
    _random = np.random.random(num)
    data = data_generator.gen_data(_random, -0.5) - 1

    param = (0.01, 0.14, 0.0554, -0.1, -3)
    func = ExampleFunc()
    fs = Functions([func])
    # x = np.linspace(-2, 2, num)
    #
    # opt_para, opt_cov = so.curve_fit(func.f, x, data, p0=param)
    #
    # y = func.f(x, *opt_para)
    # plt.plot(x, y, label="y")
    # plt.plot(x, data, label="data")
    # plt.legend()
    # plt.show()
