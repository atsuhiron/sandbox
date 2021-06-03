from typing import Tuple
import numpy as np
from sb.particle.particle_profiles import ParticleProfile
from sb.particle.field_profile import FieldProfile
from sb.particle.field_profile import WallType
import sb.particle.physical_eq as physical_eq


class ParticleDensityDistributionInitializer:
    @staticmethod
    def random(shape: Tuple[int, int], amp: float) -> np.ndarray:
        return np.random.random(shape) * amp


class ParticleField:
    def __init__(self,
                 p_prof: ParticleProfile,
                 f_prof: FieldProfile,
                 density: np.ndarray,
                 temp: np.ndarray):
        self.p_prof = p_prof
        self.f_prof = f_prof

        if density.shape != self.f_prof.field_shape:
            sh_str = " {}, {}".format(density.shape, self.f_prof.field_shape)
            assert False, "The shape of initial density and one of field profile is inconsistent:" + sh_str
        if temp.shape != self.f_prof.field_shape:
            sh_str = " {}, {}".format(temp.shape, self.f_prof.field_shape)
            assert False, "The shape of initial temperature and one of field profile is inconsistent:" + sh_str

        self.num_dens = density
        self.temp = temp

    def calc_force_p(self) -> np.ndarray:
        # ベクトル量なので、(shape_y, shape_y, 4) の形にしたい
        # 最後の 4 は [up, right, left, bottom] = [up, right, -right, -top]
        press = physical_eq.get_pres(self.num_dens, self.temp)  # shape == (x, y)
        _d_press_y = press[1:] - press[:-1]
        _d_press_x = press[:, 1:] - press[:, :-1]

        d_press = np.zeros(shape=self.f_prof.field_shape + (4,), dtype=np.float64)
        d_press[1:, :, 0] = _d_press_y
        d_press[:-1, :, 1] = -_d_press_y
        d_press[:, 1:, 2] = _d_press_x
        d_press[:, :-1, 3] = -_d_press_x

        if self.f_prof.wall_type_top == WallType.OPEN:
            d_press[0, :, 0] = press[0]
        if self.f_prof.wall_type_bottom == WallType.OPEN:
            d_press[-1, :, 1] = press[-1]
        if self.f_prof.wall_type_left == WallType.OPEN:
            d_press[:, 0, 2] = press[:, 0]
        if self.f_prof.wall_type_right == WallType.OPEN:
            d_press[:, -1, 3] = press[:, -1]
        return -d_press * self.p_prof.moveability

    def calc_force_g(self) -> np.ndarray:
        pass
