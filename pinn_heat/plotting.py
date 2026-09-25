"""Shared figure style and field plots."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

BLUE, ORANGE, GREY = "#2a6fdb", "#e8743b", "#6b7280"


def field_panels(t, x, T_pred, T_true, obs=None, title_pred="PINN prediction"):
    """Three heatmaps over (t, x): prediction, exact solution, absolute error."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    extent = [t.min(), t.max(), x.min(), x.max()]
    vmin, vmax = T_true.min(), T_true.max()
    panels = [(T_pred, title_pred, "viridis"), (T_true, "Exact solution", "viridis"),
              (np.abs(T_pred - T_true), "Absolute error", "magma")]
    for ax, (Z, title, cmap) in zip(axes, panels):
        kw = {"vmin": vmin, "vmax": vmax} if cmap == "viridis" else {}
        im = ax.imshow(Z.T, origin="lower", extent=extent, aspect="auto", cmap=cmap, **kw)
        ax.set(title=title, xlabel="t", ylabel="x")
        ax.grid(False)
        fig.colorbar(im, ax=ax, format="%.0e" if cmap == "magma" else None)
    if obs is not None:
        axes[0].scatter(obs["t"], obs["x"], s=14, c="white", edgecolors="black", linewidths=0.6,
                        label="observations")
        axes[0].legend(loc="upper right", fontsize=8)
    return fig
