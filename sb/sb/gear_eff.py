import numpy as np
import matplotlib.pyplot as plt


ratios = np.array([3, 9, 18, 27, 36, 45])
stages = np.array([2, 3, 3, 4, 3, 4])
powers = np.array([8.62, 17.62, 29.9, 41.70, 47.2, 54.48])


plt.plot(ratios, powers, "o")
plt.show()