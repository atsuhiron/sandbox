from typing import Union
import numpy as np
import matplotlib.pyplot as plt


def serial(r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 4 + 2 * np.sqrt(2) + r + 2 * np.sqrt(1 + r*r)


def alternating(r: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    return 1 + 2 * np.sqrt(5) + r + 2 * np.sqrt(1.25 + r*r + r)


rs = np.linspace(0, 4, 50)
plt.plot(rs, serial(rs), label="serial")
plt.plot(rs, alternating(rs), label="alternating")
plt.legend()
plt.show()