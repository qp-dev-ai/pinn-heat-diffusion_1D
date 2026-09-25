"""Physics-Informed Neural Network for the 1D heat equation (JAX)."""

import jax

jax.config.update("jax_enable_x64", True)  # float64 everywhere: needed for accurate second derivatives

from .data import collocation_points, noisy_observations  # noqa: E402
from .model import beta_from_alpha, init_mlp  # noqa: E402
from .physics import exact_solution, sensitivity  # noqa: E402
from .pinn import get_alpha, make_loss, predict, residuals, train_lbfgs  # noqa: E402
