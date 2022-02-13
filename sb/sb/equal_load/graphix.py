from typing import List

import numpy as np
import matplotlib.pyplot as plt

from sb.equal_load.load import ILoad


def _to_numpy_array(loads: List[List[ILoad]]) -> np.ndarray:
    max_length = max([len(load_set) for load_set in loads])
    carrier_num = len(loads)

    arr = np.zeros((max_length, carrier_num))
    for ci in range(carrier_num):
        for li in range(len(loads[ci])):
            arr[li, ci] = loads[ci][li].get_load()
    return arr


def show_load(loads: List[List[ILoad]]):
    load_arr = _to_numpy_array(loads)

    prev_height = np.zeros(len(loads))
    x = list(range(len(loads)))
    for li in range(len(load_arr)):
        plt.bar(x, load_arr[li], bottom=prev_height)
        prev_height += load_arr[li]
    plt.show()


def show_2load(loads1: List[List[ILoad]], loads2: List[List[ILoad]]):
    for loads, sp in zip([loads1, loads2], [211, 212]):
        plt.subplot(sp)
        load_arr = _to_numpy_array(loads)

        prev_height = np.zeros(len(loads))
        x = list(range(len(loads)))
        for li in range(len(load_arr)):
            plt.bar(x, load_arr[li], bottom=prev_height)
            prev_height += load_arr[li]
    plt.show()