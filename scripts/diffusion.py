"""Heat diffusing along a 1D metal bar, animated for several thermal diffusivities alpha.

Exact solution only, no PINN: illustrates how alpha controls the diffusion speed.

    python scripts/diffusion.py
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinn_heat.animate import diffusion_bar_gif  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.18, 0.5, 1.5])
p.add_argument("--t_max", type=float, default=1.5)
p.add_argument("--n_frames_per_alpha", type=int, default=60)
p.add_argument("--fps", type=int, default=25)
args = p.parse_args()

out = ROOT / "figures" / "diffusion_alpha.gif"
diffusion_bar_gif(out, args.alphas, t_max=args.t_max, n_frames_per_alpha=args.n_frames_per_alpha, fps=args.fps)
print(f"Saved {out}")
