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
        if tk == 0:
            continue  # skip t=0: fixed sinusoidal IC, nothing to "measure"
        x_k = rng.uniform(0.05 * L, 0.95 * L, n_per_time)
        Tc_k = exact_solution(x_k, tk, alpha, L)
        sigma = noise_level * np.std(Tc_k)
        ts.append(np.full(n_per_time, tk))
        xs.append(x_k)
        T_clean.append(Tc_k)
        T_obs.append(Tc_k + sigma * rng.standard_normal(n_per_time))
    return {"t": np.concatenate(ts), "x": np.concatenate(xs), "T_obs": np.concatenate(T_obs),
            "T_clean": np.concatenate(T_clean), "sigma": noise_level}


obs = profile_observations(PROFILE_TIMES, args.n_per_time, args.alpha_true, args.noise, L, seed=args.data_seed)
print(f"Inverse problem (simple) | alpha_true = {args.alpha_true} | {len(obs['t'])} observations "
      f"| noise {100 * args.noise:.0f}% | alpha_init = {args.alpha_init}")

rec = ProfileRecorder(L)
params, hist, alpha_hat = fit_inverse(obs, alpha_init=args.alpha_init, maxiter=args.maxiter,
                                      callback=rec, callback_iters=snapshot_iterations(args.maxiter))
if rec.iters[-1] != len(hist["loss"]) - 1:
    rec(len(hist["loss"]) - 1, params)

err = 100 * abs(alpha_hat - args.alpha_true) / args.alpha_true
print(f"\nalpha_hat = {alpha_hat:.4f} (true {args.alpha_true}, relative error {err:.2f}%)")

out = ROOT / "figures"
inverse_single_gif(out / "inverse_simple_training.gif", rec, args.alpha_true, hist, obs)
print(f"Figures saved to {out}")
