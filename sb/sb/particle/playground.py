import numpy as np
import matplotlib.pyplot as plt

sh = (5, 5)
#dens = np.arange(sh[0]*sh[1], dtype=np.float64).reshape(sh)
dens = np.zeros(sh, dtype=np.float64); dens[2, 2] = 10
temp = np.ones(sh)
k_move = 0.1

def calc_d_dens(dens, temp, k_move):
    pres = dens * temp

    d_pres_y_u = np.zeros(sh)
    d_pres_y_l = np.zeros(sh)
    d_pres_x_l = np.zeros(sh)
    d_pres_x_r = np.zeros(sh)

    _y_d = pres[1:] - pres[:-1]
    _x_d = pres[:, 1:] - pres[:, :-1]
    d_pres_y_u[1:] = _y_d
    d_pres_y_l[:-1] = -_y_d
    d_pres_x_l[:, 1:] = _x_d
    d_pres_x_r[:, :-1] = -_x_d

    is_free_wall = False
    if is_free_wall:
        d_pres_y_u[0] = pres[0]
        d_pres_y_l[-1] = pres[-1]
        d_pres_x_l[:, 0] = pres[:, 0]
        d_pres_x_r[:, -1] = pres[:, -1]

    d_dens = np.array([d_pres_y_u, d_pres_y_l, d_pres_x_l, d_pres_x_r]).sum(axis=0)
    return d_dens * (-k_move)


dens_log = np.zeros(200)
for ii in range(200):
    d_dens = calc_d_dens(dens, temp, k_move)
    dens += d_dens
    dens_log[ii] = dens.mean()

plt.plot(dens_log)
plt.show()
