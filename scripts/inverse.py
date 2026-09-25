"""Inverse problem: recover the thermal diffusivity alpha from sparse, noisy data.

Three training variants are compared on the same dataset:
  1. alpha = alpha_floor + beta^2, unweighted data loss
  2. alpha = alpha_floor + beta^2, two-stage sensitivity-weighted data loss
  3. alpha = softplus(beta), unweighted: collapses to the trivial solution alpha -> 0

    python scripts/inverse.py
    python scripts/inverse.py --alpha_true 0.18 --noise 0.1
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from pinn_heat import exact_solution, noisy_observations, predict  # noqa: E402
from pinn_heat.animate import ProfileRecorder, inverse_comparison_gif  # noqa: E402
from pinn_heat.inverse import fit_inverse, sensitivity_weights  # noqa: E402
from pinn_heat.pinn import snapshot_iterations  # noqa: E402
from pinn_heat.plotting import BLUE, GREY, ORANGE, field_panels, plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--alpha_true", type=float, default=0.5)
p.add_argument("--alpha_init", type=float, default=1.0)
p.add_argument("--n_obs", type=int, default=25)
p.add_argument("--noise", type=float, default=0.20, help="noise std as a fraction of std(T)")
p.add_argument("--data_seed", type=int, default=0)
p.add_argument("--maxiter", type=int, default=3000)
args = p.parse_args()

L, T_MAX = 1.0, 1.5
obs = noisy_observations(args.n_obs, args.alpha_true, args.noise, T_MAX, L, seed=args.data_seed)
print(f"Inverse problem | alpha_true = {args.alpha_true} | {len(obs['t'])}/{args.n_obs} observations kept "
      f"| noise {100 * args.noise:.0f}% | alpha_init = {args.alpha_init}")
fit = dict(alpha_init=args.alpha_init, maxiter=args.maxiter, verbose=False)

print("1/3 beta^2, unweighted ...")
_, h_plain, a_plain = fit_inverse(obs, **fit)

print("2/3 beta^2, sensitivity-weighted (weights from the estimate of run 1) ...")
rec = ProfileRecorder(L, times=np.array([0.0, 0.4, 0.8]))
params, h_sens, a_sens = fit_inverse(obs, weights=sensitivity_weights(obs, a_plain, L), callback=rec,
                                     callback_iters=snapshot_iterations(args.maxiter), **fit)
if rec.iters[-1] != len(h_sens["loss"]) - 1:
    rec(len(h_sens["loss"]) - 1, params)

print("3/3 softplus, unweighted ...")
_, h_soft, a_soft = fit_inverse(obs, alpha_param="softplus", **fit)

print()
for name, a in [("beta^2, unweighted", a_plain), ("beta^2, sensitivity-weighted", a_sens), ("softplus, unweighted", a_soft)]:
    print(f"  {name:30s} alpha = {a:.5g}   (relative error {100 * abs(a - args.alpha_true) / args.alpha_true:.2f}%)")

runs = [
    {"label": "softplus: collapse to alpha = 0", "color": ORANGE, "hist": h_soft},
    {"label": f"beta^2, unweighted: {a_plain:.4f}", "color": GREY, "hist": h_plain},
    {"label": f"beta^2, sensitivity-weighted: {a_sens:.4f}", "color": BLUE, "hist": h_sens},
]

# Figures
out = ROOT / "figures"
t = np.linspace(0, T_MAX, 200)
x = np.linspace(0, L, 200)
Tg, Xg = np.meshgrid(t, x, indexing="ij")
T_pred = np.array(predict(params["nn"], jnp.array(Tg.ravel()), jnp.array(Xg.ravel()), L)).reshape(Tg.shape)
T_true = exact_solution(Xg, Tg, args.alpha_true, L)
fig = field_panels(t, x, T_pred, T_true, obs=obs, title_pred="PINN reconstruction")
fig.suptitle(f"Inverse problem (sensitivity-weighted) - {len(obs['t'])} observations, {100 * args.noise:.0f}% noise - "
             f"alpha = {a_sens:.4f} (true {args.alpha_true})", y=1.04)
fig.savefig(out / "inverse_field.png")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
ax1.axhline(args.alpha_true, color="black", ls=":", label=f"true alpha = {args.alpha_true}")
for r in runs:
    ax1.plot(np.arange(1, len(r["hist"]["alpha"]) + 1), r["hist"]["alpha"], color=r["color"], label=r["label"])
    ax2.loglog(r["hist"]["loss_phys"], color=r["color"])
    ax2.loglog(r["hist"]["loss_data"], color=r["color"], ls="--")
ax1.set(xscale="log", xlabel="L-BFGS iteration", ylabel="alpha", title="Estimated diffusivity")
ax1.legend(fontsize=8)
ax2.plot([], [], color="0.3", label="physics loss")
ax2.plot([], [], color="0.3", ls="--", label="data MSE")
ax2.set(xlabel="L-BFGS iteration", ylabel="loss", title="Training losses")
ax2.legend(fontsize=8)
fig.savefig(out / "inverse_alpha_losses.png")

inverse_comparison_gif(out / "inverse_training.gif", rec, runs, args.alpha_true, obs,
                       title=f"Inverse problem: alpha and T(x, t) from {len(obs['t'])} points with "
                             f"{100 * args.noise:.0f}% noise, starting from alpha = {args.alpha_init}")
print(f"Figures saved to {out}")
