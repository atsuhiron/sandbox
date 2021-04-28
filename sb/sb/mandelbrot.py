import warnings
from multiprocessing import Pool, Process
import itertools

import numpy as np
import matplotlib.pyplot as plt
# import scipy.optimize as so
from tqdm import tqdm

warnings.resetwarnings()
warnings.simplefilter('error')


def apx_func(x, amp, base, offset):
    return amp * np.power(base, x) + offset


def _recurrence(xy: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.array([xy[0] * xy[0] - xy[1] * xy[1] + a, 2 * np.product(xy) + b], dtype=np.float64)


def _remove_nan(array: np.ndarray) -> np.ndarray:
    filt_nan = np.isnan(array)
    filt_inf = np.isinf(array)
    return array[~np.logical_or(filt_nan, filt_inf)]


def calc_steep(init_x: float, init_y: float, max_num: int) -> int:
    rec = np.zeros(max_num, dtype=np.float64) + np.nan
    rec[0] = 0
    xy = np.zeros(2, dtype=np.float64)
    for ii in range(1, max_num):
        try:
            xy = _recurrence(xy, init_x, init_y)
            rec[ii] = np.linalg.norm(xy)
        except RuntimeWarning:
            rec[ii] = np.nan
            break

    rec = _remove_nan(rec)

    if len(rec) < 3:
        return np.nan
    return len(rec)


def calc_steep_wrap(args):
    return calc_steep(*args)


def assign_result(res_in: int, tq_obj: tqdm) -> int:
    tq_obj.update(1)
    return res_in


if __name__ == "__main__":
    pool = Pool(12)
    chunk_size = 128

    repeat = 255
    resolution = 2**15
    rr = 1.8
    field_x = np.linspace(-rr, rr, resolution, dtype=np.float64)
    field_y = np.linspace(-rr, rr, resolution, dtype=np.float64)
    # res = np.zeros((resolution, resolution), dtype=np.uint8)
    arguments = itertools.product(field_x, field_y, [repeat])

    # res = np.array(pool.map(calc_steep_wrap, arguments, chunk_size), dtype=np.uint8)
    #pool.close()

    res = np.zeros(resolution * resolution, dtype=np.uint8)
    c = 0
    with tqdm(total=resolution**2) as tq:
        #res = np.array([assign_result(reslet, tq) for reslet in pool.imap(calc_steep_wrap, arguments, chunk_size)])
        for reslet in pool.imap(calc_steep_wrap, arguments, chunk_size):
            res[c] = reslet
            c += 1
            tq.update(1)
        pool.close()

    res = res.reshape((resolution, resolution), order="F")
    plt.imshow(res)
    plt.show()