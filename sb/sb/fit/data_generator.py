import numpy as np
import scipy.ndimage as sn


def gen_data(origin_data: np.ndarray, r: float) -> np.ndarray:
    if origin_data.ndim == 1:
        length = len(origin_data)
        smoothed = np.zeros((length, length))
        for ii in range(length):
            smoothed[ii] = sn.gaussian_filter1d(origin_data, ii+1, truncate=5.0)
        coefs = np.power(np.arange(1, length+1), r)
        coefs = coefs / np.sum(coefs)
        coefs = coefs[:, np.newaxis]
        return np.sum(smoothed * coefs, axis=0)
    elif origin_data.ndim == 2:
        size = origin_data.shape
        order = min(size)
        smoothed = np.zeros((order, size[0], size[1]))
        for ii in range(order):
            smoothed[ii] = sn.gaussian_filter(origin_data, ii+1, truncate=5.0)
        coefs = np.power(np.arange(1, order + 1), r)
        coefs = coefs / np.sum(coefs)
        coefs = coefs[:, np.newaxis, np.newaxis]
        return np.sum(smoothed * coefs, axis=0)
    else:
        return origin_data


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(10)
    dd_1d = np.random.random(240)
    rs = [-0.3, 0, 0.3]
    for rr in rs:
        res_1d = gen_data(dd_1d, rr)
        plt.plot(res_1d, label=str(rr))

    plt.legend()
    plt.show()

    dd_2d = np.random.random((120, 180))
    res_2d = gen_data(dd_2d, -0)
    plt.imshow(res_2d)
    plt.colorbar()
    plt.show()
