"""Does more data help? Error on alpha vs. number of observations.

For each (alpha_true, N, dataset seed), the inverse problem is solved unweighted
and with two-stage sensitivity weighting (stage 1 = the unweighted fit).
A second set of unweighted runs keeps the negative measurements, to test whether
discarding them causes the systematic underestimation of alpha.

    python scripts/data_size_study.py      # ~1 h on a laptop CPU
"""

import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pinn_heat import noisy_observations  # noqa: E402
from pinn_heat.inverse import fit_inverse, sensitivity_weights  # noqa: E402
from pinn_heat.plotting import BLUE, GREY, ORANGE, plt  # noqa: E402

ALPHAS = [0.18, 0.5]
N_OBS = [10, 25, 50, 100, 250]
SEEDS = [0, 1, 2]
NOISE, ALPHA_INIT = 0.10, 1.0
N_KEEP_NEGATIVE = [25, 250]  # dataset sizes for the "keep negative measurements" test

csv = ROOT / "results" / "data_size_study.csv"
csv.parent.mkdir(exist_ok=True)
rows = []


def record(alpha, n, seed, obs, method, a_hat):
    rows.append({"alpha_true": alpha, "n_obs": n, "data_seed": seed, "n_obs_kept": len(obs["t"]),
                 "method": method, "alpha_hat": a_hat, "rel_error_pct": 100 * abs(a_hat - alpha) / alpha,
                 "signed_error_pct": 100 * (a_hat - alpha) / alpha})
    pd.DataFrame(rows).to_csv(csv, index=False)


for alpha, n, seed in itertools.product(ALPHAS, N_OBS, SEEDS):
    obs = noisy_observations(n, alpha, NOISE, seed=seed)
    _, _, a1 = fit_inverse(obs, alpha_init=ALPHA_INIT, verbose=False)
    _, _, a2 = fit_inverse(obs, alpha_init=ALPHA_INIT, weights=sensitivity_weights(obs, a1), verbose=False)
    record(alpha, n, seed, obs, "unweighted", a1)
    record(alpha, n, seed, obs, "sensitivity-weighted", a2)
    print(f"alpha={alpha} N={n:3d} seed={seed} | unweighted {a1:.4f} | weighted {a2:.4f}", flush=True)

for alpha, n, seed in itertools.product(ALPHAS, N_KEEP_NEGATIVE, SEEDS):
    obs = noisy_observations(n, alpha, NOISE, seed=seed, drop_negative=False)
    _, _, a = fit_inverse(obs, alpha_init=ALPHA_INIT, verbose=False)
    record(alpha, n, seed, obs, "unweighted, negatives kept", a)
    print(f"alpha={alpha} N={n:3d} seed={seed} | negatives kept {a:.4f}", flush=True)

df = pd.DataFrame(rows)
print("\nSigned error on alpha (%), mean over seeds\n",
      df.groupby(["alpha_true", "n_obs", "method"])["signed_error_pct"].mean().unstack().round(2))

fig, axes = plt.subplots(1, len(ALPHAS), figsize=(10, 3.6), sharey=True, constrained_layout=True)
styles = [("unweighted", GREY, "o"), ("sensitivity-weighted", BLUE, "o"), ("unweighted, negatives kept", ORANGE, "s")]
for ax, alpha in zip(axes, ALPHAS):
    for method, color, marker in styles:
        d = df[(df.alpha_true == alpha) & (df.method == method)]
        if d.empty:
            continue
        ax.scatter(d.n_obs, d.rel_error_pct, color=color, s=14, alpha=0.5, marker=marker)
        m = d.groupby("n_obs")["rel_error_pct"].mean()
        ax.plot(m.index, m.values, color=color, marker=marker, label=method)
    n_ref = np.array(N_OBS, dtype=float)
    ref = df[(df.alpha_true == alpha) & (df.method == "unweighted") & (df.n_obs == N_OBS[0])].rel_error_pct.mean()
    ax.plot(n_ref, ref * np.sqrt(n_ref[0] / n_ref), "k:", lw=1, label=r"$\propto 1/\sqrt{N}$")
    ax.set(xscale="log", yscale="log", xlabel="number of observations N", title=f"alpha_true = {alpha}")
axes[0].set_ylabel("relative error on alpha (%)")
axes[0].legend(fontsize=8)
fig.suptitle(f"Effect of the amount of data ({int(100 * NOISE)}% noise, {len(SEEDS)} datasets per point)", y=1.05)
fig.savefig(ROOT / "figures" / "data_size_study.png")
print(f"\nSaved {csv} and figures/data_size_study.png")
