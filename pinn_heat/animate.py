"""Training animations: temperature profiles converging during L-BFGS."""

import jax.numpy as jnp
import matplotlib.cm as cm
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .physics import exact_solution
from .pinn import predict
from .plotting import BLUE, ORANGE, plt

PROFILE_TIMES = np.array([0.0, 0.15, 0.4, 0.8, 1.5])


class ProfileRecorder:
    """Callback for `train_lbfgs` storing T(x, t_k) at a few fixed times."""

    def __init__(self, L=1.0, times=PROFILE_TIMES, n_x=120):
        self.L, self.times = L, times
        self.x = np.linspace(0, L, n_x)
        Tg, Xg = np.meshgrid(times, self.x, indexing="ij")
        self._t, self._x = jnp.array(Tg.ravel()), jnp.array(Xg.ravel())
        self.iters, self.profiles = [], []

    def __call__(self, it, params):
        T = np.array(predict(params["nn"], self._t, self._x, self.L)).reshape(len(self.times), -1)
        self.iters.append(it)
        self.profiles.append(T)


def training_gif(path, rec, alpha_true, history, curve="loss", obs=None, title="", fps=10, hold=20):
    """Left: PINN profiles (solid) vs exact solution (dashed) [+ observations].
    Right: training curve (loss, or alpha against its true value).
    """
    colors = cm.viridis(np.linspace(0, 0.9, len(rec.times)))
    T_exact = exact_solution(rec.x[None, :], rec.times[:, None], alpha_true, rec.L)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.25, 1]},
                                   constrained_layout=True)
    fig.suptitle(title, fontsize=11)

    lines = []
    for k, (tk, c) in enumerate(zip(rec.times, colors)):
        ax1.plot(rec.x, T_exact[k], ls="--", lw=1.2, color="0.35")
        lines.append(ax1.plot(rec.x, rec.profiles[0][k], color=c, lw=2.2, label=f"t = {tk:g}")[0])
    if obs is not None:
        ax1.scatter(obs["x"], obs["T_obs"], c=obs["t"], cmap="viridis", vmin=0, vmax=rec.times.max() / 0.9,
                    s=22, edgecolors="black", linewidths=0.5, zorder=5, label="noisy data")
    ax1.plot([], [], ls="--", color="0.35", label="exact")
    ax1.set(xlabel="x", ylabel="T(x, t)", ylim=(-0.15, 1.15), title="Temperature profiles")
    ax1.legend(fontsize=7.5, loc="upper right", ncol=2)

    its = np.arange(1, len(history[curve]) + 1)
    y = history[curve]
    (trace,) = ax2.plot([], [], color=BLUE, lw=2)
    (dot,) = ax2.plot([], [], "o", color=BLUE)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1, its[-1] * 1.2)
    if curve == "alpha":
        ax2.axhline(alpha_true, color=ORANGE, ls="--", lw=1.5, label=f"true alpha = {alpha_true}")
        ax2.set(ylabel="alpha", title="Estimated diffusivity")
        ax2.set_ylim(min(y.min(), alpha_true) / 1.5, max(y.max(), alpha_true) * 1.5)
        ax2.legend(fontsize=8)
    else:
        ax2.set(ylabel="loss", title="Training loss")
        ax2.set_ylim(y.min() / 3, y.max() * 3)
    ax2.set_xlabel("L-BFGS iteration")
    label = ax2.text(0.03, 0.05, "", transform=ax2.transAxes, fontsize=9,
                     bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    frames = list(range(len(rec.iters))) + [len(rec.iters) - 1] * hold

    def draw(f):
        it = rec.iters[f]
        for k, ln in enumerate(lines):
            ln.set_ydata(rec.profiles[f][k])
        trace.set_data(its[: it + 1], y[: it + 1])
        dot.set_data([its[it]], [y[it]])
        txt = f"iteration {it + 1}"
        if curve == "alpha":
            txt += f"\nalpha = {y[it]:.4f}"
        label.set_text(txt)
        return lines + [trace, dot, label]

    FuncAnimation(fig, draw, frames=frames, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
