"""PINN losses and L-BFGS training loop."""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LBFGS

from .model import alpha_from_beta, temperature


def pde_residual(layers, alpha, t, x, L=1.0):
    """R = dT/dt - alpha d2T/dx2 at a single point, via automatic differentiation."""
    T_t = jax.grad(temperature, argnums=1)(layers, t, x, L)
    T_xx = jax.grad(jax.grad(temperature, argnums=2), argnums=2)(layers, t, x, L)
    return T_t - alpha * T_xx


residuals = jax.vmap(pde_residual, in_axes=(None, None, 0, 0, None))
predict = jax.vmap(temperature, in_axes=(None, 0, 0, None))


def get_alpha(params, alpha_fixed=None):
    """Known alpha (forward problem) or trainable alpha (inverse problem)."""
    return alpha_fixed if alpha_fixed is not None else alpha_from_beta(params["beta"])


def make_loss(tx_col, obs=None, weights=None, alpha_fixed=None, lambda_data=1.0, L=1.0):
    """Total loss  L = L_phys + lambda_data * L_data.

    L_phys : mean squared PDE residual on the collocation points.
    L_data : (weighted) mean squared error on the observations, if any:
             sum_i w_i (T_i - T_obs_i)^2 / sum_i w_i.
    """
    t_col, x_col = jnp.asarray(tx_col[:, 0]), jnp.asarray(tx_col[:, 1])
    if obs is not None:
        t_d, x_d, T_d = (jnp.asarray(obs[k]) for k in ("t", "x", "T_obs"))
        w = jnp.ones_like(T_d) if weights is None else jnp.asarray(weights)

    def loss(params):
        alpha = get_alpha(params, alpha_fixed)
        loss_phys = jnp.mean(residuals(params["nn"], alpha, t_col, x_col, L) ** 2)
        if obs is None:
            return loss_phys
        err2 = (predict(params["nn"], t_d, x_d, L) - T_d) ** 2
        loss_data = jnp.sum(w * err2) / jnp.sum(w)
        return loss_phys + lambda_data * loss_data

    return loss


def snapshot_iterations(maxiter, n_frames=60):
    """Log-spaced iterations: training changes fast early and slowly later."""
    return set(np.unique(np.geomspace(1, maxiter, n_frames).astype(int)) - 1) | {0}


def train_lbfgs(loss, params, maxiter=3000, tol=1e-9, log_every=250, alpha_fixed=None, verbose=True,
                callback=None, callback_iters=()):
    """Minimize `loss` with L-BFGS (jaxopt), recording loss and alpha history.

    If given, `callback(it, params)` is called at the iterations in `callback_iters`
    (used to record snapshots for the training animations).
    """
    solver = LBFGS(fun=loss, maxiter=maxiter, tol=tol, history_size=10, linesearch="zoom")
    state = solver.init_state(params)
    update = jax.jit(solver.update)

    history = {"loss": [], "alpha": []}
    t0 = time.time()
    for it in range(maxiter):
        params, state = update(params, state)
        history["loss"].append(float(state.value))
        history["alpha"].append(float(get_alpha(params, alpha_fixed)))
        if callback is not None and it in callback_iters:
            callback(it, params)
        if verbose and (it % log_every == 0 or it == maxiter - 1):
            print(f"  iter {it:5d} | loss {history['loss'][-1]:.3e} | alpha {history['alpha'][-1]:.5f}")
        if not np.isfinite(history["loss"][-1]) or float(state.error) < tol:
            break
    history = {k: np.array(v) for k, v in history.items()}
    history["time_s"] = time.time() - t0
    return params, history
