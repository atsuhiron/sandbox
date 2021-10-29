from typing import Tuple
import numpy as np


def format_float(val: float) -> str:
    return str(val).replace("-", "m").replace(".", "")


class StaticParams:
    repeat = 255
    center = np.array([0.00332150318384375, 0.8423827224726562])
    resolution = 2 ** 10
    chunk_size = 2 ** 8
    frame_num = 30*10  # 30 FPS * 5 sec
    sess_name = "x{0}_y{1}_res{2}_frame{3}".format(format_float(center[0]), format_float(center[1]), resolution, frame_num)
    path_base = "sb/mandelbrot/"
    path_imgs = path_base + "images/" + sess_name + "/"
    path_array = path_base + "arrays/" + sess_name + ".npy"
    img_name_template = path_imgs + "fig_{:0>5}.png"
    _tmp_range = 0.000000001

    @classmethod
    def get_arr_shape(cls) -> Tuple[int, int, int]:
        return cls.frame_num, cls.resolution, cls.resolution