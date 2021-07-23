import numpy as np


def get_pres(density: np.ndarray, temperature: np.ndarray) -> np.ndarray:
    """
    EoS
    P = nRT
    """
    return density * temperature * 8.314462

G = 1e-6
