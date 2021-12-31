# erosion
# convey
# sediment
# upheaval

from typing import Tuple

from tqdm import tqdm
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


def gen_noise_2d(size: Tuple[int, int], power: float, max_scale: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    origin = np.random.random(size)
    noise = np.zeros(size)

    max_scale = min(max_scale, *size)
    coef_arr = np.power(np.arange(1, max_scale + 1), power).astype(np.float64)
    coef_arr /= coef_arr.sum()
    for s, p in zip(range(1, max_scale + 1), coef_arr):
        noise += sn.gaussian_filter(origin, s) * p
    return noise


def calc_d_1d(array: np.ndarray) -> np.ndarray:
    #  (f(x+h) - f(x-h)) / 2h
    size = len(array)
    d_arr = np.zeros(size)
    d_arr[1:-1] = (array[2:] - array[:-2]) * 0.5
    d_arr[0] = array[1] - array[0]
    d_arr[-1] = array[-1] - array[-2]
    return d_arr


def _calc_d_2d_x(array: np.ndarray) -> np.ndarray:
    grad_x = np.zeros_like(array)
    grad_x[1:-1] = (array[2:] - array[:-2]) * 0.5
    grad_x[0] = array[1] - array[0]
    grad_x[-1] = array[-1] - array[-2]
    return grad_x


def _calc_d_2d_y(array: np.ndarray) -> np.ndarray:
    grad_y = np.zeros_like(array)
    grad_y[:, 1:-1] = (array[:, 2:] - array[:, :-2]) * 0.5
    grad_y[:, 0] = array[:, 1] - array[:, 0]
    grad_y[:, -1] = array[:, -1] - array[:, -2]
    return grad_y


def calc_d_2d(array: np.ndarray) -> np.ndarray:
    size = array.shape
    grad = np.zeros((2, ) + size)

    grad[0] = _calc_d_2d_x(array)
    grad[1] = _calc_d_2d_y(array)
    return grad


def calc_d2_2d(array: np.ndarray) -> np.ndarray:
    grad_x2 = _calc_d_2d_x(_calc_d_2d_x(array))
    grad_y2 = _calc_d_2d_y(_calc_d_2d_y(array))
    return grad_x2 +grad_y2


# def calc_d2_1d(array: np.ndarray) -> np.ndarray:
#     # (f(x+h) + f(x-h) - 2*f(x))
#     size = len(array)
#     d2_arr = np.zeros(size)
#     d2_arr[1:-1] = array[2:] + array[:-2] - 2*array[1:-1]
#     return d2_arr


def calc_erosion_1d(array: np.ndarray, c_inst: float, c_ste: float) -> np.ndarray:
    steep = np.abs(calc_d_1d(array)) * c_ste
    instable = calc_d_1d(calc_d_1d(array)) * c_inst
    ret = steep - instable
    ret[ret < 0] = 0
    return ret


def calc_erosion_2d(array: np.ndarray, c_inst: float, c_ste: float) -> np.ndarray:
    steep = np.linalg.norm(calc_d_2d(array), axis=0) * c_ste
    instable = calc_d2_2d(array) * c_inst
    ret = steep - instable
    ret[ret < 0] = 0
    return ret


def calc_sediment_1d(array: np.ndarray, c_sed: float) -> np.ndarray:
    sed = calc_d_1d(calc_d_1d(array)) * c_sed
    ret = sed
    ret[ret < 0] = 0
    return ret


def calc_sediment_2d(array: np.ndarray, c_sed: float) -> np.ndarray:
    sed = calc_d2_2d(array) * c_sed
    ret = sed
    ret[ret < 0] = 0
    return ret


if __name__ == "__main__":
    # 1D
    # _size = 240
    # x = np.arange(_size)
    # teran = gen_noise_1d(_size, 0.2, 200, 35385)
    # es = []
    # for ii in range(59):
    #     _ero = calc_erosion_1d(teran, 0.2, 0.4)
    #     _sed = calc_sediment_1d(teran, _ero.sum()*50)
    #     diff = _sed - _ero
    #     diff = sn.gaussian_filter1d(diff, 1, truncate=3.0)
    #     teran = teran + diff
    #     if ii % 4 == 0 or ii < 4:
    #         #plt.plot(teran, color=cm.bone(ii/60.0))
    #         pass
    # plt.subplot(2,1,1)
    # plt.plot(teran)
    # plt.subplot(2,1,2)
    # plt.plot(calc_d_1d(calc_d_1d(teran)))
    # plt.show()

    # 2D
    import os
    _size = (120, 180)
    imshow_d = {"vmin": 0.4, "vmax": 0.51}
    teran = gen_noise_2d(_size, -0.4, 100, 111111)
    # plt.title("0")
    # plt.imshow(teran, **imshow_d)
    # plt.colorbar()
    # plt.savefig("sb/ecsufig/fig_{}.png".format(str(0).zfill(2)))
    # plt.clf()

    for ii in tqdm(range(1, 61)):
        _ero = calc_erosion_2d(teran, 0.2, 0.4)
        _sed = calc_sediment_2d(teran, 0.5)

        diff = sn.gaussian_filter(_sed - _ero, 1, truncate=3.0)
        teran += diff
    #     plt.title(str(ii))
    #     plt.imshow(teran, **imshow_d)
    #     plt.colorbar()
    #     plt.savefig("sb/ecsufig/fig_{}.png".format(str(ii).zfill(2)))
    #     plt.clf()
    #
    # os.system("ffmpeg -r 5 -i sb/ecsufig/fig_%2d.png -vcodec h264 -r 5 sb/mov.mp4")