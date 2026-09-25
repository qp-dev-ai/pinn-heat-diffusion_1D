"""Fully connected network with hard-constrained initial/boundary conditions."""

import jax
import jax.numpy as jnp
import numpy as np


def glorot_uniform(key, n_in, n_out):
    """W ~ U(-a, a) with a = sqrt(6 / (n_in + n_out)), zero biases."""
    limit = np.sqrt(6.0 / (n_in + n_out))
    W = jax.random.uniform(key, (n_in, n_out), minval=-limit, maxval=limit, dtype=jnp.float64)
    return {"W": W, "b": jnp.zeros((n_out,), dtype=jnp.float64)}


def init_mlp(key, width=16, depth=3):
    """MLP (t, x) -> scalar with `depth` hidden layers of `width` neurons."""
    sizes = [2] + [width] * depth + [1]
    keys = jax.random.split(key, len(sizes) - 1)
    return [glorot_uniform(k, n_in, n_out) for k, n_in, n_out in zip(keys, sizes[:-1], sizes[1:])]


def mlp(layers, tx):
    """Raw network output N(t, x); `tx` has shape (..., 2)."""
    h = tx
    for layer in layers[:-1]:
        h = jnp.tanh(h @ layer["W"] + layer["b"])
    return h @ layers[-1]["W"] + layers[-1]["b"]


def temperature(layers, t, x, L=1.0):
    """Hard-constrained ansatz for a single point (t, x):

        T(x, t) = sin(pi x / L) + t x (L - x) N(t, x)

    The initial condition and both Dirichlet boundary conditions are satisfied
    exactly for any network weights, so no IC/BC penalty terms are needed.
    """
    n = mlp(layers, jnp.stack([t, x]))[0]
    return jnp.sin(jnp.pi * x / L) + t * x * (L - x) * n


# The diffusivity is trained through an auxiliary variable beta:
#   "square"   : alpha = alpha_floor + beta^2   (default)
#   "softplus" : alpha = softplus(beta)
#   "linear"   : alpha = beta                   (no reparameterization)
# With "square", d(alpha)/d(beta) = 2 beta shrinks as alpha -> 0, which slows the
# drift of alpha toward the trivial solution alpha = 0 early in training, while
# the network has not yet fitted the data.
ALPHA_FLOOR = 1e-12
ALPHA_PARAMS = ("square", "softplus", "linear")


def alpha_from_beta(beta, param="square"):
    if param == "square":
        return ALPHA_FLOOR + beta**2
    if param == "softplus":
        return jax.nn.softplus(beta)
    if param == "linear":
        return beta
    raise ValueError(f"unknown alpha parameterization: {param}")


def beta_from_alpha(alpha, param="square"):
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    if param == "square":
        return jnp.sqrt(jnp.maximum(alpha - ALPHA_FLOOR, 0.0))
    if param == "softplus":
        return jnp.log(jnp.expm1(alpha))
    if param == "linear":
        return alpha
    raise ValueError(f"unknown alpha parameterization: {param}")
