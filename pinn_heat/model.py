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


# Positivity of the diffusivity is enforced by the reparameterization
# alpha = alpha_floor + beta^2, with beta the trainable variable.
ALPHA_FLOOR = 1e-12


def alpha_from_beta(beta):
    return ALPHA_FLOOR + beta**2


def beta_from_alpha(alpha):
    return jnp.sqrt(jnp.maximum(alpha - ALPHA_FLOOR, 0.0))
