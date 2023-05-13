import dataclasses
import time
import numpy as np
import numba
import matplotlib.pyplot as plt


@dataclasses.dataclass
class HenonMapParam:
    first: float
    second: float
    a: float = 1.4
    b: float = 0.3


@numba.jit("f8(f8,f8,f8,f8)", nopython=True)
def henon_map_float(x_1: float, x_2: float, a: float, b: float) -> float:
    return 1 - a * x_1**2 + b * x_2


@numba.jit("f8[:](f8[:],f8,f8,i8)", nopython=True)
def get_seq_core(arr: np.ndarray, a: float, b: float, length: int) -> np.ndarray:
    for i in range(2, length):
        arr[i] = henon_map_float(arr[i-1], arr[i-2], a, b)
    return arr


def get_seq(hmp: HenonMapParam, length: int) -> np.ndarray:
    arr = np.zeros(length, dtype=np.float64)
    arr[0] = hmp.first
    arr[1] = hmp.second
    return get_seq_core(arr, hmp.a, hmp.b, length)


def draw_return_map(seq: np.ndarray):
    bef = seq[:-1]
    aft = seq[1:]
    plt.plot(bef, aft, "o", markersize=5)
    plt.show()


if __name__ == "__main__":
    _hmp = HenonMapParam(0.5, 0.28)

    s = time.time()
    sequence = get_seq(_hmp, 1_000_000*2)
    elapsed = time.time() - s
    print(elapsed)

    draw_return_map(sequence)
