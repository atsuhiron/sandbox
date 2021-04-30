import os
import shutil
import warnings
from multiprocessing import Pool
import itertools
from typing import Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.resetwarnings()
warnings.simplefilter('error')


def format_float(val: float) -> str:
    return str(val).replace("-", "m").replace(".", "")


class StaticParams:
    repeat = 255
    center = np.array([-0.13997576390625, 0.650401699609375])
    resolution = 2 ** 10
    chunk_size = 2 ** 8
    frame_num = 30*10  # 30 FPS * 5 sec
    sess_name = "x{0}_y{1}_res{2}_frame{3}".format(format_float(center[0]), format_float(center[1]), resolution, frame_num)
    path_base = "sb/mandelbrot/"
    path_imgs = path_base + "images/" + sess_name + "/"
    path_array = path_base + "arrays/" + sess_name + ".npy"
    img_name_template = path_imgs + "fig_{:0>5}.png"
    _tmp_range = 0.000000002

    @classmethod
    def get_arr_shape(cls) -> Tuple[int, int, int]:
        return cls.frame_num, cls.resolution, cls.resolution


class UtilCalc:
    @staticmethod
    def calc_shift(new_center: np.ndarray) -> np.ndarray:
        pixel_shift = (new_center - StaticParams.resolution/2) * np.array([1, -1])
        coord_shift = pixel_shift * StaticParams._tmp_range / StaticParams.resolution * 2
        return StaticParams.center + coord_shift


def _recurrence(xy: np.ndarray, a: float, b: float) -> np.ndarray:
    return np.array([xy[0] * xy[0] - xy[1] * xy[1] + a, 2 * xy[0] * xy[1] + b], dtype=np.float64)


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


def calc_steep_wrap_with_index(args):
    return args[0], calc_steep(*args[1])


def assign_result(res_in: int, tq_obj: tqdm) -> int:
    tq_obj.update(1)
    return res_in


def gen_field(f_range: float, f_center: np.ndarray, f_resolution: int) -> Tuple[
    Union[np.ndarray, Tuple[np.ndarray, Optional[float]]], Union[np.ndarray, Tuple[np.ndarray, Optional[float]]]]:
    fieldx = np.linspace(f_center[0] - f_range, f_center[0] + f_range, f_resolution, dtype=np.float64)
    fieldy = np.linspace(f_center[1] - f_range, f_center[1] + f_range, f_resolution, dtype=np.float64)
    return fieldx, fieldy


def calc_set(p_pool: Pool, index: int, f_x: np.ndarray, f_y: np.ndarray):
    arguments = itertools.product(f_x, f_y, [StaticParams.repeat])
    res = np.zeros(StaticParams.resolution * StaticParams.resolution, dtype=np.uint8)
    c = 0
    with tqdm(total=StaticParams.resolution ** 2, desc="{0:0>3}/{1}".format(index+1, StaticParams.frame_num)) as tq:
        # res = np.array([assign_result(reslet, tq) for reslet in pool.imap(calc_steep_wrap, arguments, chunk_size)])
        for reslet in p_pool.imap(calc_steep_wrap, arguments, StaticParams.chunk_size):
            res[c] = reslet
            c += 1
            tq.update(1)
    return res.reshape((StaticParams.resolution, StaticParams.resolution), order="F")[::-1]


def calc_set_unordered(p_pool: Pool, index: int, f_x: np.ndarray, f_y: np.ndarray):
    arguments = enumerate(itertools.product(f_x, f_y, [StaticParams.repeat]))
    res = np.zeros(StaticParams.resolution * StaticParams.resolution, dtype=np.uint8)
    with tqdm(total=StaticParams.resolution ** 2, desc="{0:0>3}/{1}".format(index+1, StaticParams.frame_num)) as tq:
        # res = np.array([assign_result(reslet, tq) for reslet in pool.imap(calc_steep_wrap, arguments, chunk_size)])
        for index, reslet in p_pool.imap_unordered(calc_steep_wrap_with_index, arguments, StaticParams.chunk_size):
            res[index] = reslet
            tq.update(1)
    return res.reshape((StaticParams.resolution, StaticParams.resolution), order="F")[::-1]


def save_fig(array: np.ndarray, name: str, scale: float):
    fig = plt.figure(figsize=(9, 9))
    plt.imshow(array)
    plt.title("scale = {:.8E}".format(scale))
    plt.tight_layout()
    plt.axis("off")
    plt.savefig(name, dpi=128)
    plt.cla()
    plt.close()


def reset_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    reset_dir(StaticParams.path_imgs)
    pool = Pool(12)
    result_array = np.zeros(StaticParams.get_arr_shape(), dtype=np.uint8)
    scales = np.logspace(np.log10(0.000000002), np.log10(2), StaticParams.frame_num)[::-1]
    for ii in range(StaticParams.frame_num):
        fields = gen_field(scales[ii], StaticParams.center, StaticParams.resolution)
        res_let = calc_set_unordered(pool, ii, *fields)
        save_fig(res_let, StaticParams.img_name_template.format(ii), scales[ii])
        result_array[ii] = res_let
        break

    np.save(StaticParams.path_array, result_array)
    pool.close()
