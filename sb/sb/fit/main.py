from typing import Union
from typing import Dict
from typing import List
from typing import Callable
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

    def f(self, *args):
        return args[1] * args[0] ** 4 + \
               args[2] * args[0] ** 3 + \
               args[3] * args[0] ** 2 + \
               args[4] * args[0] + \
               args[5]


class ExampleFuncSin(base_function.BaseFunction):
    @property
    def func_name(self) -> str:
        return "sin"

    @property
    def argument_names(self) -> List[str]:
        return ["amp", "freq"]

    def f(self, *args):
        return args[1] * np.sin(args[2] * args[0])


class Functions:
    def __init__(self, funcs: List[base_function.BaseFunction]):
        self.funcs = funcs
        self._set_function_index()
        self.arg_dict = self._gen_arg_dict(self.funcs)

    def f(self, *args):
        variable = args[0]
        parameter = args[1:]
        self._assign_args(*parameter)

        res_values = []
        for i, f_name in enumerate(self.arg_dict):
            funclet_args = [variable] + [self.arg_dict[f_name][a_name].get_value() for a_name in self.arg_dict[f_name]]
            res_values.append(self.funcs[i].f(*funclet_args))
        return sum(res_values)

    def set_param_config(self, func: Union[str, int], param: Union[str, int],
                         status: argument.ArgStatus = None, value: float = None, d_func: Callable = None):
        if type(func) is int:
            try:
                f_name = self.funcs[func].get_unique_name()
            except IndexError:
                print("Max function index is {}".format(len(self.funcs)))
                return
            f_index = func
        else:
            f_name = func
            try:
                f_index = [ff.get_unique_name() for ff in self.funcs].index(func)
            except ValueError:
                print("Available function names are {}".format([ff.get_unique_name() for ff in self.funcs]))
                return

        if type(param) is int:
            try:
                p_name = self.funcs[f_index].argument_names[param]
            except IndexError:
                print("Max parameter index is {}".format(len(self.funcs[f_index].argument_names)))
                return
        else:
            p_name = param
            if p_name not in self.funcs[f_index].argument_names:
                print("Available parameter names are {}".format(self.funcs[f_index].argument_names))
                return

        target_arg = self.arg_dict[f_name][p_name]
        if (status is argument.ArgStatus.FREE) or (status is argument.ArgStatus.FREEZE):
            target_arg.set_status(status)
            self._set_param_config_free_or_freeze(target_arg, value, d_func)
        elif status is argument.ArgStatus.DEPEND:
            self._set_param_config_depend(target_arg, value, d_func)
        else:
            cur_status = target_arg.get_status()
            if (cur_status is argument.ArgStatus.FREE) or (cur_status is argument.ArgStatus.FREEZE):
                self._set_param_config_free_or_freeze(target_arg, value, d_func)
            elif cur_status is argument.ArgStatus.DEPEND:
                if value is not None:
                    print("Value is ignored.")
                if d_func is not None:
                    target_arg.set_depend_function(d_func)

    def show_param_config(self):
        pass

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

    @staticmethod
    def _set_param_config_free_or_freeze(target_arg: argument.Argument, value: float, d_func: Callable):
        if value is not None:
            target_arg.set_value(value)
        if d_func is not None:
            print("Depending function is ignored.")

    @staticmethod
    def _set_param_config_depend(target_arg: argument.Argument, value: float, d_func: Callable):
        if d_func is None:
            print("Depending function (d_func) is required.")
            return
        target_arg.set_status(argument.ArgStatus.DEPEND)
        target_arg.set_depend_function(d_func)
        if value is not None:
            print("Value is ignored.")


if __name__ == "__main__":
    import importlib
    importlib.reload(base_function)
    importlib.reload(argument)

    num = 128
    np.random.seed(213)
    _random = np.random.random(num)
    data = data_generator.gen_data(_random, -0.5) - 1

    init_param = (0.01, 0.14, 0.0554, -0.1, -3)
    func_1 = ExampleFunc()
    fs = Functions([func_1])

    x = np.linspace(-2, 2, num)
    opt_para, opt_cov = so.curve_fit(fs.f, x, data, p0=init_param)

    y = fs.f(x, *opt_para)
    plt.plot(x, y, label="y")
    plt.plot(x, data, label="data")
    plt.legend()
    plt.show()
