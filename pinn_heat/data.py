"""Collocation points and synthetic noisy observations."""

import numpy as np
from scipy.stats import qmc

from .physics import exact_solution


def collocation_points(n, t_max=1.5, L=1.0, seed=0):
    """Latin Hypercube Sampling of n points (t, x) in [0, t_max] x [0, L]."""
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n)
    return qmc.scale(unit, [0.0, 0.0], [t_max, L])


def noisy_observations(n, alpha, noise_level, t_max=1.5, L=1.0, seed=0, drop_negative=True):
    """Random observations of the exact solution with additive Gaussian noise.

    The noise standard deviation is `noise_level * std(T_clean)`. Observations
    with a negative measured temperature can be discarded, since they lie below
    the noise floor and are physically inconsistent with this problem.

    Returns a dict with arrays t, x, T_obs, T_clean and the noise std `sigma`.
    """
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, t_max, n)
    x = rng.uniform(0.0, L, n)
    T_clean = exact_solution(x, t, alpha, L)
    sigma = noise_level * np.std(T_clean)
    T_obs = T_clean + sigma * rng.standard_normal(n)

    keep = T_obs >= 0.0 if drop_negative else np.ones(n, dtype=bool)
    return {"t": t[keep], "x": x[keep], "T_obs": T_obs[keep], "T_clean": T_clean[keep], "sigma": sigma}
