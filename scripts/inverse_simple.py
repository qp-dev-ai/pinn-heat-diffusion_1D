"""Inverse problem: a single, straightforward run (no parameterization pathologies,
no sensitivity-weighting stage -- just data + physics, unweighted).

    python scripts/inverse_simple.py
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_heat import noisy_observations  # noqa: E402
from pinn_heat.animate import ProfileRecorder, training_gif  # noqa: E402
from pinn_heat.inverse import fit_inverse  # noqa: E402
from pinn_heat.pinn import snapshot_iterations  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--alpha_true", type=float, default=0.5)
p.add_argument("--alpha_init", type=float, default=0.3)
p.add_argument("--n_obs", type=int, default=25)
p.add_argument("--noise", type=float, default=0.10, help="noise std as a fraction of std(T)")
p.add_argument("--data_seed", type=int, default=0)
p.add_argument("--maxiter", type=int, default=3000)
args = p.parse_args()

L, T_MAX = 1.0, 1.5
obs = noisy_observations(args.n_obs, args.alpha_true, args.noise, T_MAX, L, seed=args.data_seed)
print(f"Inverse problem (simple) | alpha_true = {args.alpha_true} | {len(obs['t'])}/{args.n_obs} observations kept "
      f"| noise {100 * args.noise:.0f}% | alpha_init = {args.alpha_init}")

rec = ProfileRecorder(L)
params, hist, alpha_hat = fit_inverse(obs, alpha_init=args.alpha_init, maxiter=args.maxiter,
                                      callback=rec, callback_iters=snapshot_iterations(args.maxiter))
if rec.iters[-1] != len(hist["loss"]) - 1:
    rec(len(hist["loss"]) - 1, params)

err = 100 * abs(alpha_hat - args.alpha_true) / args.alpha_true
print(f"\nalpha_hat = {alpha_hat:.4f} (true {args.alpha_true}, relative error {err:.2f}%)")

out = ROOT / "figures"
training_gif(out / "inverse_simple_alpha.gif", rec, args.alpha_true, hist, curve="alpha", obs=obs)
training_gif(out / "inverse_simple_loss.gif", rec, args.alpha_true, hist, curve="loss", obs=obs)
print(f"Figures saved to {out}")
