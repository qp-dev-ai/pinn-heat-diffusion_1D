"""Inverse problem: recover the thermal diffusivity alpha from sparse, noisy data.

    python scripts/inverse.py                       # unweighted data loss
    python scripts/inverse.py --weighting sensitivity   # two-stage sensitivity weighting
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from pinn_heat import exact_solution, noisy_observations, predict  # noqa: E402
from pinn_heat.animate import ProfileRecorder, training_gif  # noqa: E402
from pinn_heat.pinn import snapshot_iterations  # noqa: E402
from pinn_heat.inverse import fit_inverse, fit_inverse_two_stage  # noqa: E402
from pinn_heat.plotting import BLUE, GREY, ORANGE, field_panels, plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--alpha_true", type=float, default=0.18)
p.add_argument("--alpha_init", type=float, default=1.0)
p.add_argument("--n_obs", type=int, default=25)
p.add_argument("--noise", type=float, default=0.10, help="noise std as a fraction of std(T)")
p.add_argument("--weighting", choices=["none", "sensitivity"], default="none")
p.add_argument("--data_seed", type=int, default=0)
p.add_argument("--maxiter", type=int, default=3000)
args = p.parse_args()

L, T_MAX = 1.0, 1.5
obs = noisy_observations(args.n_obs, args.alpha_true, args.noise, T_MAX, L, seed=args.data_seed)
print(f"Inverse problem | alpha_true = {args.alpha_true} | {len(obs['t'])}/{args.n_obs} observations kept "
      f"| noise {100 * args.noise:.0f}% | weighting: {args.weighting}")

rec = ProfileRecorder(L)
snap = {"callback": rec, "callback_iters": snapshot_iterations(args.maxiter)}
if args.weighting == "none":
    params, hist, alpha_hat = fit_inverse(obs, alpha_init=args.alpha_init, maxiter=args.maxiter, **snap)
    histories = [hist]
else:
    params, histories, alpha_hat, _ = fit_inverse_two_stage(obs, alpha_init=args.alpha_init,
                                                            maxiter=args.maxiter, **snap)
if rec.iters[-1] != len(histories[-1]["loss"]) - 1:
    rec(len(histories[-1]["loss"]) - 1, params)

err = abs(alpha_hat - args.alpha_true) / args.alpha_true
print(f"\nalpha estimated = {alpha_hat:.5f}  (true {args.alpha_true}, relative error {100 * err:.2f}%)")

# Figures
t = np.linspace(0, T_MAX, 200)
x = np.linspace(0, L, 200)
Tg, Xg = np.meshgrid(t, x, indexing="ij")
T_pred = np.array(predict(params["nn"], jnp.array(Tg.ravel()), jnp.array(Xg.ravel()), L)).reshape(Tg.shape)
T_true = exact_solution(Xg, Tg, args.alpha_true, L)

out = ROOT / "figures"
tag = f"_{args.weighting}"
fig = field_panels(t, x, T_pred, T_true, obs=obs, title_pred="PINN reconstruction")
fig.suptitle(f"Inverse problem - {len(obs['t'])} observations, {100 * args.noise:.0f}% noise - "
             f"alpha = {alpha_hat:.4f} (true {args.alpha_true})", y=1.04)
fig.savefig(out / f"inverse_field{tag}.png")

fig, ax = plt.subplots(figsize=(5.5, 3.4))
colors = [GREY, BLUE] if len(histories) == 2 else [BLUE]
labels = ["stage 1 (unweighted)", "stage 2 (sensitivity-weighted)"] if len(histories) == 2 else ["PINN estimate"]
for h, c, lab in zip(histories, colors, labels):
    ax.plot(h["alpha"], color=c, label=lab)
ax.axhline(args.alpha_true, color=ORANGE, ls="--", label="true alpha")
ax.set(xlabel="L-BFGS iteration", ylabel="alpha", yscale="log", title="Convergence of the diffusivity")
ax.legend(fontsize=8)
fig.savefig(out / f"inverse_alpha{tag}.png")
stage = " (stage 2, sensitivity-weighted)" if args.weighting == "sensitivity" else ""
training_gif(out / f"inverse_training{tag}.gif", rec, args.alpha_true, histories[-1], curve="alpha", obs=obs,
             title=f"Inverse problem{stage}: alpha and T(x, t) from {len(obs['t'])} points with "
                   f"{100 * args.noise:.0f}% noise, starting from alpha = {args.alpha_init}")
print(f"Figures saved to {out}")
