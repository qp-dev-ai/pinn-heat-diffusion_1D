"""Forward problem: alpha is known, first solved from physics alone, then with a
handful of noisy temperature measurements added to the loss.

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

from pinn_heat.animate import PROFILE_TIMES, ProfileRecorder, training_gif  # noqa: E402
from pinn_heat.pinn import snapshot_iterations  # noqa: E402
from pinn_heat import collocation_points, exact_solution, init_mlp, make_loss, predict, train_lbfgs  # noqa: E402
from pinn_heat.plotting import BLUE, field_panels, plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--alpha", type=float, default=0.18)
p.add_argument("--n_col", type=int, default=500)
p.add_argument("--n_per_time", type=int, default=4)
p.add_argument("--noise", type=float, default=0.15)
p.add_argument("--width", type=int, default=16)
p.add_argument("--depth", type=int, default=3)
p.add_argument("--maxiter", type=int, default=3000)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

L, T_MAX = 1.0, 1.5
out = ROOT / "figures"


def profile_observations(times, n_per_time, alpha, noise_level, L, seed):
    """A few noisy measurements at each of the plotted profile times, so the data
    points sit close to the very curves shown in the figure (rather than at
    arbitrary times where no curve is drawn to compare them against).
    """
    rng = np.random.default_rng(seed)
    ts, xs, T_obs, T_clean = [], [], [], []
    for tk in times:
        if tk == 0:
            continue  # skip t=0: fixed sinusoidal IC, nothing to "measure"
        x_k = rng.uniform(0.1 * L, 0.9 * L, n_per_time)
        Tc_k = exact_solution(x_k, tk, alpha, L)
        sigma = noise_level * np.std(Tc_k)
        ts.append(np.full(n_per_time, tk))
        xs.append(x_k)
        T_clean.append(Tc_k)
        T_obs.append(Tc_k + sigma * rng.standard_normal(n_per_time))
    return {"t": np.concatenate(ts), "x": np.concatenate(xs), "T_obs": np.concatenate(T_obs),
            "T_clean": np.concatenate(T_clean), "sigma": noise_level}


def run(label, obs):
    """Train with alpha fixed and known, physics only if obs is None else physics + data."""
    params = {"nn": init_mlp(jax.random.PRNGKey(args.seed), args.width, args.depth)}
    tx_col = collocation_points(args.n_col, T_MAX, L, seed=args.seed)
    loss = make_loss(tx_col, obs=obs, alpha_fixed=args.alpha, L=L)

    n_obs = 0 if obs is None else len(obs["t"])
    print(f"Forward problem ({label}) | alpha = {args.alpha} | {args.n_col} collocation points, "
          f"{n_obs} observations")
    rec = ProfileRecorder(L)
    params, hist = train_lbfgs(loss, params, maxiter=args.maxiter, alpha_fixed=args.alpha,
                               callback=rec, callback_iters=snapshot_iterations(args.maxiter))
    if rec.iters[-1] != len(hist["loss"]) - 1:
        rec(len(hist["loss"]) - 1, params)

    t = np.linspace(0, T_MAX, 200)
    x = np.linspace(0, L, 200)
    Tg, Xg = np.meshgrid(t, x, indexing="ij")
    T_pred = np.array(predict(params["nn"], jnp.array(Tg.ravel()), jnp.array(Xg.ravel()), L)).reshape(Tg.shape)
    T_true = exact_solution(Xg, Tg, args.alpha, L)
    rel_l2 = np.linalg.norm(T_pred - T_true) / np.linalg.norm(T_true)

    print(f"{len(hist['loss'])} L-BFGS iterations in {hist['time_s']:.1f} s")
    print(f"Final loss         : {hist['loss'][-1]:.2e}")
    print(f"Relative L2 error   : {rel_l2:.2e}\n")
    return rec, hist, t, x, T_pred, T_true, rel_l2


# Physics only, no data
rec, hist, t, x, T_pred, T_true, rel_l2 = run("no data", obs=None)
fig = field_panels(t, x, T_pred, T_true)
fig.suptitle(f"Forward problem (alpha = {args.alpha}, no data) - relative L2 error {rel_l2:.1e}", y=1.04)
fig.savefig(out / "forward_field.png")

fig, ax = plt.subplots(figsize=(5, 3.2))
ax.semilogy(hist["loss"], color=BLUE)
ax.set(xlabel="training iteration", ylabel="loss (physics)", title="Forward problem - training")
fig.savefig(out / "forward_loss.png")
training_gif(out / "forward_training.gif", rec, args.alpha, hist, curve="loss")

# Physics + a few noisy measurements, close to the profile curves shown in the figure
obs = profile_observations(PROFILE_TIMES, args.n_per_time, args.alpha, args.noise, L, seed=args.seed)
rec_d, hist_d, t_d, x_d, T_pred_d, T_true_d, rel_l2_d = run("with data", obs=obs)
training_gif(out / "forward_data_training.gif", rec_d, args.alpha, hist_d, curve="loss", obs=obs)

print(f"Figures saved to {out}")
