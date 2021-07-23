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

        self.shape: Tuple[int, int] = self.f_prof.field_shape
        self.g_pot_filter = self._gen_gravity_potential_filter()
        self.num_dens = density
        self.temp = temp

    def calc_force_p(self) -> np.ndarray:  # (x, y, 4)
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

    def calc_force_g(self) -> np.ndarray:  # (x, y, 4)
        mass_dens = self.p_prof.mass * self.num_dens
        g_pot = ss.convolve2d(mass_dens, self.g_pot_filter, mode="same", boundary="fill", fillvalue=0.0)
        d_g_pot = self._calc_delta(g_pot)
        return d_g_pot

    def calc_d_dens(self, dens: np.ndarray = None) -> np.ndarray:  # (x, y)
        moving_ratio = 0.5
        if dens is None:
            dens = self.num_dens  # (x, y)

        #  calc gravity
        d_g_pot = self.calc_force_g()
        d_dens_grav_neg = d_g_pot
        d_dens_grav_neg[d_dens_grav_neg > 0] = 0
        d_dens_grav_pos = np.abs(d_dens_grav_neg)
        d_dens = np.zeros(self.shape, dtype=np.float64)

        # The part of increase due to inflow.
        inflow = d_dens_grav_pos * dens[:, :, np.newaxis] * moving_ratio
        d_dens[:-1] += inflow[1:, :, 0]  # up
        d_dens[1:] += inflow[:-1, :, 1]  # down
        d_dens[:, :-1] += inflow[:, 1:, 2]  # left
        d_dens[:, 1:] += inflow[:, :-1, 3]  # right

        # The part of decrease due to outflow.
        outflow = np.sum(d_dens_grav_neg, axis=2) * dens * moving_ratio
        d_dens += outflow

        #  calc pressure
        d_dens += np.sum(self.calc_force_p(), axis=2)

        return d_dens

    def _calc_delta(self, array2d: np.ndarray) -> np.ndarray:
        _d_array2d_y = array2d[1:] - array2d[:-1]
        _d_array2d_x = array2d[:, 1:] - array2d[:, :-1]

        d_array2d = np.zeros(shape=self.shape + (4,), dtype=np.float64)
        d_array2d[1:, :, 0] = _d_array2d_y  # top
        d_array2d[:-1, :, 1] = -_d_array2d_y  # bottom
        d_array2d[:, 1:, 2] = _d_array2d_x  # left
        d_array2d[:, :-1, 3] = -_d_array2d_x  #right
        return d_array2d

    def _gen_gravity_potential_filter(self) -> np.ndarray:
        filt_shape = list(self.shape)
        if filt_shape[0] % 2 == 0:
            filt_shape[0] += 1
        if filt_shape[1] % 2 == 0:
            filt_shape[1] += 1

        dist_x, dist_y = np.meshgrid(self._get_distance_array(filt_shape[1]),
                                     self._get_distance_array(filt_shape[0]))
        inv_dist = np.sqrt(dist_x * dist_x + dist_y * dist_y)
        inv_dist[inv_dist == 0] = 0.5
        return -physical_eq.G / inv_dist

    @staticmethod
    def _get_distance_array(num: int) -> np.ndarray:
        return np.linspace((1 - num) / 2, (num - 1) / 2, num)
