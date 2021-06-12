from multiprocessing import Pool
from typing import Tuple
import numpy as np
import tqdm

from sb.particle.particle_profiles import ParticleProfile
from sb.particle.field_profile import FieldProfile
import sb.particle.particle_field as pf
import sb.particle.graphic_path_manager as gpm

import importlib
importlib.reload(pf)
importlib.reload(gpm)

#  https://qiita.com/shotoyoo/items/43a49439899334c6516e
#  確認する


def draw_thread(arg: Tuple[gpm.GraphicPathManager, int, np.ndarray, dict]):
    gpm_man, index, array, imshow_kw = arg
    gpm_man.save_frame(index, array, imshow_kw)


if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    man = gpm.GraphicPathManager("test", 3)
    # init_dens = np.array([
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 1, 0, 1, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    # ], dtype=np.float64)
    # init_dens = np.array([
    #     [0, 0, 1, 0, 2, 2, 0, 0],
    #     [1, 0, 0, 0, 0, 1, 1, 0],
    #     [0, 0, 0, 0, 0, 2, 2, 0],
    #     [0, 0, 0, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 1, 0, 0],
    #     [1, 0, 0, 0, 0, 0, 2, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0],
    # ], dtype=np.float64)
    init_dens = np.random.random((128, 64)) * 10
    init_temp = np.ones_like(init_dens, dtype=np.float64) * 80
    f_prof = FieldProfile(init_dens.shape)
    p_prof = ParticleProfile(0, 50)

    particle_field = pf.ParticleField(p_prof, f_prof, init_dens, init_temp)

    NN = 500
    result_arr = np.zeros((NN, ) + init_dens.shape, dtype=np.float64)
    for ii in tqdm.tqdm(range(NN), desc="CALC"):
        #d_grav = particle_field.calc_force_g()
        result_arr[ii] = particle_field.num_dens
        d_dens = particle_field.calc_d_dens(None)
        particle_field.num_dens += d_dens

    imshow_dict = {"cmap": "gist_ncar", "vmin": None, "vmax": None}
    p_pool = Pool(10)
    args_list = [(man, ii, result_arr[ii], imshow_dict) for ii in range(NN)]
    with tqdm.tqdm(total=NN) as tt:
        for _ in p_pool.imap_unordered(draw_thread, args_list, 4):
            tt.update(1)
    p_pool.close()
    tt.close()
    man.gen_mov()
