"""1D heat equation: exact solution and parameter sensitivity.

    dT/dt = alpha * d2T/dx2,   x in [0, L],  t in [0, t_max]
    T(x, 0) = sin(pi x / L),   T(0, t) = T(L, t) = 0
"""

import numpy as np


def exact_solution(x, t, alpha, L=1.0):
    """T(x, t) = sin(pi x / L) * exp(-alpha (pi / L)^2 t)."""
    k2 = (np.pi / L) ** 2
    return np.sin(np.pi * x / L) * np.exp(-alpha * k2 * t)


def sensitivity(x, t, alpha, L=1.0):
    """S_alpha = dT/dalpha = -(pi / L)^2 t T(x, t).

    Measures how much information an observation at (x, t) carries about alpha.
    It vanishes at t = 0 (initial profile does not depend on alpha) and at late
    times (the signal has decayed), and peaks at t = 1 / (alpha (pi / L)^2).
    """
    k2 = (np.pi / L) ** 2
    return -k2 * t * exact_solution(x, t, alpha, L)
