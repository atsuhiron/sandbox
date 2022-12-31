import numpy as np
import matplotlib.pyplot as plt


def gen_random_data(num: int, feature_num: int) -> np.ndarray:
    return np.random.random((num, feature_num)) - 0.5


def gen_linear_data(num: int, feature_num: int, snr: float) -> np.ndarray:
    x_core = np.random.random(num) - 0.5
    params = np.random.random((feature_num, 2))
    x_raw = np.array([x_core*param[0] + param[1] for param in params])
    x_raw = np.transpose(x_raw)
    noise = np.random.random(size=x_raw.shape)
    return x_raw + noise * snr


def get_s_mat(sigma_arr: np.ndarray, total: int) -> np.ndarray:
    diag_mat = np.diag(sigma_arr)
    if len(sigma_arr) >= total:
        return diag_mat

    foot = np.zeros((total - len(sigma_arr), len(sigma_arr)))
    return np.r_[diag_mat, foot]


if __name__ == "__main__":
    feat_num = 8
    data_num = 20
    d_lin = gen_linear_data(data_num, feat_num, 0.1)
    d_ran = gen_random_data(data_num, feat_num)

    u_lin, s_lin, v_lin = np.linalg.svd(d_lin)
    for n in range(feat_num):
        s_copy = s_lin.copy()
        s_copy[n+1:] = 0.0
        smat = get_s_mat(s_copy, data_num)
        plt.subplot(2, feat_num, n + 1)
        plt.imshow(u_lin @ (smat @ v_lin))
        plt.axis("off")

    u_ran, s_ran, v_ran = np.linalg.svd(d_ran)
    for n in range(feat_num):
        s_copy = s_ran.copy()
        s_copy[n + 1:] = 0.0
        smat = get_s_mat(s_copy, data_num)
        plt.subplot(2, feat_num, n + 1 + feat_num)
        plt.imshow(u_ran @ (smat @ v_ran))
        plt.axis("off")

    plt.show()