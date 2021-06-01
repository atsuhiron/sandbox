from typing import Tuple
import numpy as np
from sb.particle.particle_profiles import ParticleProfile
from sb.particle.field_profile import FieldProfile
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

        self.dens = density
        self.temp = temp

    def calc_force_p(self) -> np.ndarray:
        # ベクトル量なので、(shape_y, shape_y, 4) の形にしたい
        # 最後の 4 は [up, right, left, bottom] = [up, right, -right, -top]
        return physical_eq.get_pres(self.dens, self.temp)
