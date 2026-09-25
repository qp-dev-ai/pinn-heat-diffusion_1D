"""Robustness study: alpha recovery vs. noise level, with and without sensitivity weighting.

Each configuration (alpha_true, noise level, dataset seed) is solved twice:
unweighted data loss, and two-stage sensitivity-weighted data loss.

    python scripts/robustness_study.py      # ~30 min on a laptop CPU
"""

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pinn_heat import noisy_observations  # noqa: E402
from pinn_heat.inverse import fit_inverse, fit_inverse_two_stage  # noqa: E402
from pinn_heat.plotting import BLUE, GREY, plt  # noqa: E402

ALPHAS = [0.18, 0.5]
NOISES = [0.05, 0.10, 0.20]
SEEDS = [0, 1, 2]
N_OBS, ALPHA_INIT = 25, 1.0

csv = ROOT / "results" / "robustness_study.csv"
csv.parent.mkdir(exist_ok=True)

rows = []
for alpha, noise, seed in itertools.product(ALPHAS, NOISES, SEEDS):
    obs = noisy_observations(N_OBS, alpha, noise, seed=seed)
    _, _, a_plain = fit_inverse(obs, alpha_init=ALPHA_INIT, verbose=False)
    _, _, a_sens, _ = fit_inverse_two_stage(obs, alpha_init=ALPHA_INIT, verbose=False)
    for method, a_hat in [("unweighted", a_plain), ("sensitivity-weighted", a_sens)]:
        rows.append({"alpha_true": alpha, "noise": noise, "data_seed": seed, "n_obs_kept": len(obs["t"]),
                     "method": method, "alpha_hat": a_hat, "rel_error_pct": 100 * abs(a_hat - alpha) / alpha})
    print(f"alpha={alpha} noise={noise:.2f} seed={seed} | unweighted {a_plain:.4f} | weighted {a_sens:.4f}")
    pd.DataFrame(rows).to_csv(csv, index=False)

df = pd.DataFrame(rows)
summary = df.groupby(["alpha_true", "noise", "method"])["rel_error_pct"].agg(["mean", "max"]).round(2)
print("\nRelative error on alpha (%) over dataset seeds\n", summary)

# Figure: relative error vs noise level, one panel per alpha_true
fig, axes = plt.subplots(1, len(ALPHAS), figsize=(10, 3.4), sharey=True, constrained_layout=True)
for ax, alpha in zip(np.atleast_1d(axes), ALPHAS):
    for method, color, dx in [("unweighted", GREY, -0.004), ("sensitivity-weighted", BLUE, 0.004)]:
        d = df[(df.alpha_true == alpha) & (df.method == method)]
        ax.scatter(d.noise + dx, d.rel_error_pct, color=color, s=18, alpha=0.7)
        m = d.groupby("noise")["rel_error_pct"].mean()
        ax.plot(m.index + dx, m.values, color=color, marker="o", label=method)
    ax.set(title=f"alpha_true = {alpha}", xlabel="noise level (fraction of std T)", xticks=NOISES)
axes[0].set_ylabel("relative error on alpha (%)")
axes[0].legend(fontsize=8)
fig.suptitle(f"Recovery of alpha from {N_OBS} noisy observations ({len(SEEDS)} datasets per point)", y=1.05)
fig.savefig(ROOT / "figures" / "robustness_study.png")
print(f"\nSaved {csv} and figures/robustness_study.png")
