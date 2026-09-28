"""Inverse problem: a single, straightforward run (no parameterization pathologies,
no sensitivity-weighting stage -- just data + physics, unweighted).

    python scripts/inverse_simple.py
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

from pinn_heat import exact_solution  # noqa: E402
from pinn_heat.animate import PROFILE_TIMES, ProfileRecorder, inverse_single_gif  # noqa: E402
from pinn_heat.inverse import fit_inverse  # noqa: E402
from pinn_heat.pinn import snapshot_iterations  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--alpha_true", type=float, default=0.3)
p.add_argument("--alpha_init", type=float, default=0.1)
p.add_argument("--n_per_time", type=int, default=25)
p.add_argument("--noise", type=float, default=0.10, help="noise std as a fraction of std(T)")
p.add_argument("--data_seed", type=int, default=0)
p.add_argument("--maxiter", type=int, default=3000)
args = p.parse_args()

L, T_MAX = 1.0, 1.5


def profile_observations(times, n_per_time, alpha, noise_level, L, seed):
    """Noisy measurements at each of the plotted profile times, so the data points sit
    close to (and visually align with) the very curves shown in the figure.
    """
    rng = np.random.default_rng(seed)
    ts, xs, T_obs, T_clean = [], [], [], []
    for tk in times:
        x_k = rng.uniform(0.05 * L, 0.95 * L, n_per_time)
        Tc_k = exact_solution(x_k, tk, alpha, L)
        sigma = noise_level * np.std(Tc_k)
        ts.append(np.full(n_per_time, tk))
        xs.append(x_k)
        T_clean.append(Tc_k)
        T_obs.append(Tc_k + sigma * rng.standard_normal(n_per_time))
    return {"t": np.concatenate(ts), "x": np.concatenate(xs), "T_obs": np.concatenate(T_obs),
            "T_clean": np.concatenate(T_clean), "sigma": noise_level}


def fit_alpha_least_squares(obs, L):
    """Classical (non-PINN) least-squares fit of the analytical solution to the data alone:
    the alpha an L2 data loss can actually be expected to recover, noise and all -- as
    opposed to the noise-free alpha_true used to generate that data in the first place.
    """
    k2 = (np.pi / L) ** 2

    def sse(alpha):
        pred = np.sin(np.pi * obs["x"] / L) * np.exp(-alpha * k2 * obs["t"])
        return np.sum((pred - obs["T_obs"]) ** 2)

    return minimize_scalar(sse, bounds=(1e-4, 5.0), method="bounded").x


obs = profile_observations(PROFILE_TIMES, args.n_per_time, args.alpha_true, args.noise, L, seed=args.data_seed)
alpha_fit = fit_alpha_least_squares(obs, L)
print(f"Inverse problem (simple) | alpha_true = {args.alpha_true} | {len(obs['t'])} observations "
      f"| noise {100 * args.noise:.0f}% | alpha_init = {args.alpha_init}")

rec = ProfileRecorder(L)
params, hist, alpha_hat = fit_inverse(obs, alpha_init=args.alpha_init, maxiter=args.maxiter,
                                      callback=rec, callback_iters=snapshot_iterations(args.maxiter))
if rec.iters[-1] != len(hist["loss"]) - 1:
    rec(len(hist["loss"]) - 1, params)

err = 100 * abs(alpha_hat - args.alpha_true) / args.alpha_true
err_fit = 100 * abs(alpha_hat - alpha_fit) / alpha_fit
print(f"\nalpha_true = {args.alpha_true}")
print(f"alpha_fit  = {alpha_fit:.4f}  (least-squares fit of the exact solution to the data alone)")
print(f"alpha_hat  = {alpha_hat:.4f}  (PINN estimate; {err:.2f}% from true, {err_fit:.2f}% from the data fit)")

out = ROOT / "figures"
inverse_single_gif(out / "inverse_simple_training.gif", rec, args.alpha_true, hist, obs, alpha_ref=alpha_fit)
print(f"Figures saved to {out}")
