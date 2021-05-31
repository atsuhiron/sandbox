
import numpy as np
from sb.particle.particle_profiles import ParticleProfile
from sb.particle.field_profile import FieldProfile


class ParticleField:
    def __init__(self, p_prof: ParticleProfile, f_prof: FieldProfile):
        self.p_prof = p_prof
        self.f_prof = f_prof