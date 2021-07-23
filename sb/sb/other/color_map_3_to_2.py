import numpy as np


def color_map_3_to_2(col: np.ndarray, channels: int = 3, boundary_proc: bool = True) -> np.ndarray:
    assert channels > 0
    t_coef = 127 - np.sum(col, axis=-1) / 3
    res = None
    if col.ndim <= 1:
        m_vec = np.ones_like(col) * t_coef
        res = col + m_vec
    else:
        ele: int = int(np.prod(col.shape[:-1]))
        t_coef = np.ravel(t_coef)  # shape = (ele,)
        m_vec_ravel = t_coef[:, np.newaxis] * np.ones((ele, channels))  # shape = (ele, channel)
        m_vec = m_vec_ravel.reshape(col.shape)
        res = col + m_vec

    if not boundary_proc:
        return res
    res = np.round(res)
    res[res > 255] = 255
    res[res < 0] = 0
    return res.astype(np.uint8)


if __name__ == "__main__":
    ar = np.array([0, 0, 0], dtype=np.uint8)
    print("{} -> {}".format(ar, color_map_3_to_2(ar)))

    ar2 = np.array([255, 0, 0], dtype=np.uint8)
    print("{} -> {}".format(ar2, color_map_3_to_2(ar2)))

    ar3 = np.array([[[175, 28, 235],
                     [237, 246, 58],
                     [129, 42, 72],
                     [5, 10, 133]],

                    [[163, 112, 228],
                     [145, 129, 234],
                     [159, 44, 39],
                     [172, 169, 208]]], dtype=np.uint8)
    print("{} \n->\n {}".format(ar3, color_map_3_to_2(ar3)))
