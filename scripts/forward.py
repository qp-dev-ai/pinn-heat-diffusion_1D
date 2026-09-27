"""Forward problem: alpha is known, the PINN reconstructs T(x, t) from physics only.

    python scripts/forward.py
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from pinn_heat.animate import ProfileRecorder, training_gif  # noqa: E402
from pinn_heat.pinn import snapshot_iterations  # noqa: E402
from pinn_heat import collocation_points, exact_solution, init_mlp, make_loss, predict, train_lbfgs  # noqa: E402
from pinn_heat.plotting import BLUE, field_panels, plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--alpha", type=float, default=0.18)
p.add_argument("--n_col", type=int, default=500)
p.add_argument("--width", type=int, default=16)
p.add_argument("--depth", type=int, default=3)
p.add_argument("--maxiter", type=int, default=3000)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

L, T_MAX = 1.0, 1.5

params = {"nn": init_mlp(jax.random.PRNGKey(args.seed), args.width, args.depth)}
tx_col = collocation_points(args.n_col, T_MAX, L, seed=args.seed)
loss = make_loss(tx_col, alpha_fixed=args.alpha, L=L)

print(f"Forward problem | alpha = {args.alpha} | {args.n_col} collocation points")
rec = ProfileRecorder(L)
params, hist = train_lbfgs(loss, params, maxiter=args.maxiter, alpha_fixed=args.alpha,
                           callback=rec, callback_iters=snapshot_iterations(args.maxiter))
if rec.iters[-1] != len(hist["loss"]) - 1:
    rec(len(hist["loss"]) - 1, params)

# Evaluation on a 200 x 200 grid
t = np.linspace(0, T_MAX, 200)
x = np.linspace(0, L, 200)
Tg, Xg = np.meshgrid(t, x, indexing="ij")
T_pred = np.array(predict(params["nn"], jnp.array(Tg.ravel()), jnp.array(Xg.ravel()), L)).reshape(Tg.shape)
T_true = exact_solution(Xg, Tg, args.alpha, L)
rel_l2 = np.linalg.norm(T_pred - T_true) / np.linalg.norm(T_true)

print(f"\n{len(hist['loss'])} L-BFGS iterations in {hist['time_s']:.1f} s")
print(f"Final physics loss : {hist['loss'][-1]:.2e}")
print(f"Relative L2 error  : {rel_l2:.2e}")
print(f"Max absolute error : {np.abs(T_pred - T_true).max():.2e}")

out = ROOT / "figures"
fig = field_panels(t, x, T_pred, T_true)
fig.suptitle(f"Forward problem (alpha = {args.alpha}) - relative L2 error {rel_l2:.1e}", y=1.04)
fig.savefig(out / "forward_field.png")

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.semilogy(hist["loss"], color=BLUE)
ax.set(xlabel="training iteration", ylabel="loss (physics)", title="Forward problem - training")
fig.savefig(out / "forward_loss.png")
training_gif(out / "forward_training.gif", rec, args.alpha, hist, curve="loss")
print(f"Figures saved to {out}")
