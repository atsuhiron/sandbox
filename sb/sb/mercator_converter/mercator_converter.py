from typing import Callable
from typing import Tuple

import numpy as np
import cv2


def cvt_x(x:float) -> float:
    return np.tan(x)


def cvt_y(y: float) -> float:
    return y


class PictureConverter:
    def __init__(self, path: str, ang_v: float, ang_h: float,
                 converter_x: Callable[[float], float], converter_y: Callable[[float], float]):
        self.path = path
        self.img: np.ndarray = cv2.imread(path)

        self.ang_v = ang_v
        self.ang_h = ang_h
        self.size: Tuple[int, int] = self.img.shape[:2]
        self.ang_arr_h, self.ang_arr_v = np.meshgrid(np.linspace(-self.ang_h/2, self.ang_h/2, self.size[1]),
                                                     np.linspace(-self.ang_v/2, self.ang_v/2, self.size[0]))

        self.cvt_x = converter_x
        self.cvt_y = converter_y

    def get_perspective(self, viewing_angle_v: float, viewing_angle_h: float):
        assert viewing_angle_v < self.ang_v
        assert viewing_angle_h < min(self.ang_h, 180)
        pass

    def _get_array(self, viewing_angle_v: float, viewing_angle_h: float) -> np.ndarray:
        """
        各ピクセルに対応するオリジナル画像の座標を返す
        :param viewing_angle_v:
        :param viewing_angle_h:
        :return:
        """
        pass
