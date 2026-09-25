"""Inverse problem: estimate alpha and T(x, t) from sparse noisy observations."""

import jax
import numpy as np

from .data import collocation_points
from .model import beta_from_alpha, init_mlp
from .physics import sensitivity
from .pinn import make_loss, train_lbfgs


def sensitivity_weights(obs, alpha_est, L=1.0, power=0.5, w_min=0.05):
    """Observation weights from the sensitivity |dT/dalpha| at the current estimate.

    w_i = max( (|S_i| / max|S|)^power , w_min )

    power = 1/2 softens the concentration around the sensitivity peak, and w_min
    keeps weakly informative observations in the loss.
    """
    s = np.abs(sensitivity(obs["x"], obs["t"], alpha_est, L))
    return np.maximum((s / (s.max() + 1e-12)) ** power, w_min)


def fit_inverse(obs, alpha_init=1.0, weights=None, alpha_param="square", n_col=500, width=16, depth=3,
                lambda_data=1.0, maxiter=3000, seed=0, t_max=1.5, L=1.0, verbose=True, **train_kw):
    """Train an inverse PINN; returns (params, history, alpha_estimate).

    Extra keyword arguments (e.g. a snapshot callback) are passed to `train_lbfgs`.
    """
    params = {"nn": init_mlp(jax.random.PRNGKey(seed), width, depth), "beta": beta_from_alpha(alpha_init, alpha_param)}
    tx_col = collocation_points(n_col, t_max, L, seed=seed)
    loss = make_loss(tx_col, obs=obs, weights=weights, alpha_param=alpha_param, lambda_data=lambda_data, L=L)
    params, hist = train_lbfgs(loss, params, maxiter=maxiter, alpha_param=alpha_param, verbose=verbose, **train_kw)
    return params, hist, hist["alpha"][-1]


def fit_inverse_two_stage(obs, verbose=True, **kw):
    """Stage 1: unweighted fit -> alpha_1.
    Stage 2: retrain with sensitivity weights evaluated at alpha_1.
    A snapshot callback, if given, only records stage 2.
    """
    stage1_kw = {k: v for k, v in kw.items() if k not in ("callback", "callback_iters")}
    _, hist1, alpha1 = fit_inverse(obs, verbose=verbose, **stage1_kw)
    w = sensitivity_weights(obs, alpha1, kw.get("L", 1.0))
    params, hist2, alpha2 = fit_inverse(obs, weights=w, verbose=verbose, **kw)
    return params, (hist1, hist2), alpha2, w
