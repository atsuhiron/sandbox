from typing import Tuple
import numpy as np
import scipy.signal as ss
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

        self.shape = self.f_prof.field_shape
        self.g_pot_filter = self._gen_gravity_potential_filter()
        self.num_dens = density
        self.temp = temp

    def calc_force_p(self) -> np.ndarray:
        # ベクトル量なので、(shape_y, shape_y, 4) の形にしたい
        # 最後の 4 は [up, right, left, bottom] = [up, right, -right, -top]
        press = physical_eq.get_pres(self.num_dens, self.temp)  # shape == (x, y)
        d_press = self._calc_delta(press)

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
        mass_dens = self.p_prof.mass * self.num_dens
        g_pot = ss.convolve2d(mass_dens, self.g_pot_filter, mode="same", boundary="fill", fillvalue=0.0)
        d_g_pot = self._calc_delta(g_pot)
        return d_g_pot

    def _calc_delta(self, array2d: np.ndarray) -> np.ndarray:
        _d_array2d_y = array2d[1:] - array2d[:-1]
        _d_array2d_x = array2d[:, 1:] - array2d[:, :-1]

        d_array2d = np.zeros(shape=self.shape + (4,), dtype=np.float64)
        d_array2d[1:, :, 0] = _d_array2d_y
        d_array2d[:-1, :, 1] = -_d_array2d_y
        d_array2d[:, 1:, 2] = _d_array2d_x
        d_array2d[:, :-1, 3] = -_d_array2d_x
        return d_array2d

    def _gen_gravity_potential_filter(self) -> np.ndarray:
        dist_x, dist_y = np.meshgrid(self._get_distance_array(self.shape[1]),
                                     self._get_distance_array(self.shape[0]))
        inv_dist = np.sqrt(dist_x * dist_x + dist_y * dist_y)
        inv_dist[inv_dist == 0] = 0.5
        return -physical_eq.G / inv_dist

    @staticmethod
    def _get_distance_array(num: int) -> np.ndarray:
        return np.linspace((1 - num) / 2, (num - 1) / 2, num)
