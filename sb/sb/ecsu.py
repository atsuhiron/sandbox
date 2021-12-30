# erosion
# convey
# sediment
# upheaval

from typing import Union
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.ndimage as sn


def gen_noise_1d(size: int, power: float, max_scale: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    origin = np.random.random(size)
    noise = np.zeros(size)

    max_scale = min(size, max_scale)
    coef_arr = np.power(np.arange(1, max_scale + 1), power).astype(np.float64)
    coef_arr /= coef_arr.sum()
    for s, p in zip(range(1, max_scale + 1), coef_arr):
        noise += sn.gaussian_filter1d(origin, s) * p
    return noise


def calc_d_1d(array: np.ndarray) -> np.ndarray:
    #  (f(x+h) - f(x-h)) / 2h
    size = len(array)
    d_arr = np.zeros(size)
    d_arr[1:-1] = (array[2:] - array[:-2]) * 0.5
    d_arr[0] = array[1] - array[0]
    d_arr[-1] = array[-1] - array[-2]
    return d_arr


def calc_d2_1d(array: np.ndarray) -> np.ndarray:
    # (f(x+h) + f(x-h) - 2*f(x))
    size = len(array)
    d2_arr = np.zeros(size)
    d2_arr[1:-1] = array[2:] + array[:-2] - 2*array[1:-1]
    return d2_arr


def calc_erosion_1d(array: np.ndarray, c_inst: float, c_ste: float) -> np.ndarray:
    steep = np.abs(calc_d_1d(array)) * c_ste
    instable = calc_d_1d(calc_d_1d(array)) * c_inst
    ret = steep - instable
    ret[ret < 0] = 0
    return ret


def calc_sediment(array: np.ndarray, c_sed: float) -> np.ndarray:
    sed = calc_d_1d(calc_d_1d(array)) * c_sed
    ret = sed
    ret[ret < 0] = 0
    return ret


if __name__ == "__main__":
    _size = 240
    x = np.arange(_size)
    teran = gen_noise_1d(_size, 0.2, 200, 35385)
    es = []
    for ii in range(59):
        _ero = calc_erosion_1d(teran, 0.2, 0.4)
        _sed = calc_sediment(teran, _ero.sum()*50)
        diff = _sed - _ero
        diff = sn.gaussian_filter1d(diff, 1)
        teran = teran + diff
        if ii % 4 == 0 or ii < 4:
            plt.plot(teran, color=cm.bone(ii/60.0))

    plt.show()