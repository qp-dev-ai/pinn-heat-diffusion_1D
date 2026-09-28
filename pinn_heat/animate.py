"""Training animations: temperature profiles converging during L-BFGS."""

import jax.numpy as jnp
import matplotlib.cm as cm
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .physics import exact_solution
from .pinn import predict
from .plotting import BLUE, GREY, ORANGE, plt

PROFILE_TIMES = np.array([0.0, 0.15, 0.4, 0.8, 1.5])


def diffusion_bar_gif(path, alphas, L=1.0, t_max=1.5, n_x=400, n_frames_per_alpha=60, hold=15, fps=25,
                       cmap="inferno"):
    """Heat diffusing along a single bar, exact solution, cycling through each alpha in turn.

    One strip is rendered as a 1D heatmap (colour = temperature); it replays the diffusion
    from t=0 for each alpha in `alphas`, one after another, with no PINN involved.
    """
    x = np.linspace(0, L, n_x)
    times = np.linspace(0, t_max, n_frames_per_alpha)
    frames = [(alpha, t) for alpha in alphas for t in times] + \
             [(alphas[-1], times[-1])] * hold

    fig, ax = plt.subplots(figsize=(7.5, 2.6), constrained_layout=True)

    T0 = exact_solution(x, 0.0, alphas[0], L)[None, :]
    im = ax.imshow(T0, aspect="auto", cmap=cmap, vmin=0, vmax=1, extent=[0, L, 0, 1])
    ax.set_yticks([])
    ax.set_xticks([0, L / 2, L], labels=["0", "L/2", "L"])
    ax.set_xlabel("x (m)")
    fig.colorbar(im, ax=ax, label="T(x, t)  [dimensionless]")

    # All three lines share ax.transAxes so they stay centred on the strip, not the whole figure.
    title = ax.text(0.5, 1.65, "Heat diffusion along a 1D bar", transform=ax.transAxes, ha="center",
                     fontsize=11)
    label_alpha = ax.text(0.5, 1.38, "", transform=ax.transAxes, ha="center", fontsize=13, color="#b85c1e")
    label_t = ax.text(0.5, 1.14, "", transform=ax.transAxes, ha="center", fontsize=10, color="0.3")

    def draw(f):
        alpha, t = frames[f]
        im.set_data(exact_solution(x, t, alpha, L)[None, :])
        label_alpha.set_text(f"$\\bf{{\\alpha = {alpha:g}}}$ m²/s")
        label_t.set_text(f"t = {t:.2f} s")
        return [im, label_alpha, label_t]

    FuncAnimation(fig, draw, frames=len(frames), blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


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


def training_gif(path, rec, alpha_true, history, curve="loss", obs=None, fps=10, hold=20):
    """Left: PINN profiles (solid) vs exact solution (dashed) [+ observations].
    Right: training curve (loss, or alpha against its true value).
    """
    colors = cm.viridis(np.linspace(0, 0.9, len(rec.times)))
    T_exact = exact_solution(rec.x[None, :], rec.times[:, None], alpha_true, rec.L)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.25, 1]},
                                   constrained_layout=True)

    lines = []
    for k, (tk, c) in enumerate(zip(rec.times, colors)):
        ax1.plot(rec.x, T_exact[k], ls="--", lw=1.2, color="0.35")
        lines.append(ax1.plot(rec.x, rec.profiles[0][k], color=c, lw=2.2, label=f"t = {tk:g} s")[0])
    if obs is not None:
        # Colour each point like its closest profile line (by time), not by raw t value:
        # rec.times is unevenly spaced, so a value-based colormap would not match the
        # rank-based line colours above.
        idx = np.argmin(np.abs(obs["t"][:, None] - rec.times[None, :]), axis=1)
        ax1.scatter(obs["x"], obs["T_obs"], c=colors[idx],
                    s=22, edgecolors="black", linewidths=0.5, zorder=5, label="noisy data")
    ax1.plot([], [], ls="--", color="0.35", label="exact")
    ax1.set(xlabel="x (m)", ylabel="T(x, t)", ylim=(-0.15, 1.15), title="Temperature profiles")
    ax1.legend(fontsize=7.5, loc="upper right", ncol=2)

    its = np.arange(1, len(history[curve]) + 1)
    y = history[curve]
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(1, its[-1] * 1.2)
    if curve == "alpha":
        (trace,) = ax2.plot([], [], color=BLUE, lw=2)
        (dot,) = ax2.plot([], [], "o", color=BLUE)
        traces = [(trace, dot, y)]
        ax2.axhline(alpha_true, color=ORANGE, ls="--", lw=1.5, label=f"true alpha = {alpha_true}")
        ax2.set(ylabel="alpha", title="Estimated diffusivity")
        ax2.set_ylim(min(y.min(), alpha_true) / 1.5, max(y.max(), alpha_true) * 1.5)
        ax2.legend(fontsize=8)
    else:
        has_data = obs is not None
        total_label = "total loss" if has_data else "total loss = physics loss"
        (trace_total,) = ax2.plot([], [], color=BLUE, lw=2, label=total_label)
        (dot_total,) = ax2.plot([], [], "o", color=BLUE)
        traces = [(trace_total, dot_total, history["loss"])]
        if has_data:
            (trace_phys,) = ax2.plot([], [], color=GREY, lw=1.5, ls="--", label="physics loss")
            (dot_phys,) = ax2.plot([], [], "o", color=GREY, ms=4)
            (trace_data,) = ax2.plot([], [], color=ORANGE, lw=1.5, ls=":", label="data loss")
            (dot_data,) = ax2.plot([], [], "o", color=ORANGE, ms=4)
            traces += [(trace_phys, dot_phys, history["loss_phys"]), (trace_data, dot_data, history["loss_data"])]
        all_y = np.concatenate([yy for *_, yy in traces])
        all_y = all_y[np.isfinite(all_y) & (all_y > 0)]
        ax2.set(ylabel="loss", title="Training loss")
        ax2.set_ylim(all_y.min() / 3, all_y.max() * 8)
        ax2.legend(fontsize=7, loc="upper right")
    ax2.set_xlabel("training iteration")
    label = ax2.text(0.03, 0.05, "", transform=ax2.transAxes, fontsize=9,
                     bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    frames = list(range(len(rec.iters))) + [len(rec.iters) - 1] * hold

    def draw(f):
        it = rec.iters[f]
        for k, ln in enumerate(lines):
            ln.set_ydata(rec.profiles[f][k])
        for trace, dot, yy in traces:
            trace.set_data(its[: it + 1], yy[: it + 1])
            dot.set_data([its[it]], [yy[it]])
        txt = f"iteration {it + 1}"
        if curve == "alpha":
            txt += f"\nalpha = {y[it]:.4f}"
        label.set_text(txt)
        return lines + [t[0] for t in traces] + [t[1] for t in traces] + [label]

    FuncAnimation(fig, draw, frames=frames, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def inverse_comparison_gif(path, rec, runs, alpha_true, obs, title="", fps=10, hold=20):
    """Inverse problem, several training variants side by side.

    Left: T(x, t_k) of the run whose snapshots are in `rec` (solid) vs exact (dashed) + data.
    Middle: alpha vs iteration for every run. Right: physics loss (solid) and data MSE (dashed).
    `runs` is a list of dicts with keys "label", "color", "hist".
    """
    colors = cm.viridis(np.linspace(0, 0.75, len(rec.times)))
    T_exact = exact_solution(rec.x[None, :], rec.times[:, None], alpha_true, rec.L)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.15, 1, 1]},
                                        constrained_layout=True)
    fig.suptitle(title, fontsize=11)

    # Profiles
    lines = []
    for k, (tk, c) in enumerate(zip(rec.times, colors)):
        ax1.plot(rec.x, T_exact[k], ls="--", lw=1.2, color="0.35")
        lines.append(ax1.plot(rec.x, rec.profiles[0][k], color=c, lw=2.4, label=f"t = {tk:g}")[0])
    # Colour each point like its closest profile line (by time), not by raw t value:
    # rec.times is unevenly spaced, so a value-based colormap would not match the
    # rank-based line colours above.
    idx = np.argmin(np.abs(obs["t"][:, None] - rec.times[None, :]), axis=1)
    ax1.scatter(obs["x"], obs["T_obs"], c=colors[idx],
                s=22, edgecolors="black", linewidths=0.5, zorder=5, label="noisy data (colour = time)")
    ax1.plot([], [], ls="--", color="0.35", label="exact")
    ax1.set(xlabel="x", ylabel="T(x, t)", ylim=(-0.2, 1.2), title=f"Temperature profiles ({runs[-1]['label']})")
    ax1.legend(fontsize=7.5, loc="upper right")

    # alpha and losses
    n_it = max(len(r["hist"]["alpha"]) for r in runs)
    its = np.arange(1, n_it + 1)
    ax2.axhline(alpha_true, color="black", ls=":", lw=1.5, label=f"true alpha = {alpha_true}")
    traces = []
    for r in runs:
        h = r["hist"]
        a = ax2.plot([], [], color=r["color"], lw=2, label=r["label"])[0]
        lp = ax3.plot([], [], color=r["color"], lw=2)[0]
        ld = ax3.plot([], [], color=r["color"], lw=1.5, ls="--")[0]
        traces.append((h, a, lp, ld))
    ax2.set(xscale="log", xlim=(1, n_it * 1.2), ylim=(-0.03, 1.05), xlabel="training iteration", ylabel="alpha",
            title="Estimated diffusivity")
    ax2.legend(fontsize=7.5, loc="upper right")
    ax3.plot([], [], color="0.3", lw=2, label="physics loss (PDE residual)")
    ax3.plot([], [], color="0.3", lw=1.5, ls="--", label="data MSE")
    all_l = np.concatenate([np.r_[r["hist"]["loss_phys"], r["hist"]["loss_data"]] for r in runs])
    all_l = all_l[np.isfinite(all_l) & (all_l > 0)]
    ax3.set(xscale="log", yscale="log", xlim=(1, n_it * 1.2), ylim=(all_l.min() / 3, all_l.max() * 3),
            xlabel="training iteration", ylabel="loss", title="Training losses")
    ax3.legend(fontsize=7.5, loc="lower left")
    label = ax2.text(0.03, 0.05, "", transform=ax2.transAxes, fontsize=9,
                     bbox=dict(boxstyle="round", fc="white", ec="0.8"))

    frames = list(range(len(rec.iters))) + [len(rec.iters) - 1] * hold

    def draw(f):
        it = rec.iters[f]
        for k, ln in enumerate(lines):
            ln.set_ydata(rec.profiles[f][k])
        for h, a, lp, ld in traces:
            j = min(it, len(h["alpha"]) - 1) + 1
            a.set_data(its[:j], h["alpha"][:j])
            lp.set_data(its[:j], h["loss_phys"][:j])
            ld.set_data(its[:j], h["loss_data"][:j])
        label.set_text(f"iteration {it + 1}")
        return lines + [label]

    FuncAnimation(fig, draw, frames=frames, blit=False).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
