# pinn-heat-diffusion_1D
Physics-Informed Neural Network for 1D heat transfer

Estimation of the temperature field and thermal diffusivity from sparse and noisy measurements using a Physics-Informed Neural Network implemented with JAX.

## Project objectives

- Solve the one-dimensional heat equation.
- Estimate an unknown thermal diffusivity.
- Study robustness to sparse and noisy measurements.
- Improve parameter identification using sensitivity-based weighting.

## Status

Work in progress. This repository currently exposes the reference JAX implementation only
(a much larger set of exploratory scripts and experiment outputs exists locally and is not
published). Everything here uses `jax.config.update("jax_enable_x64", True)`, a `tanh`
network, and hard-coded initial/boundary conditions built directly into the network output
(no soft IC/BC penalty terms). Training is L-BFGS only (`jaxopt.LBFGS`), run directly from
the initialized weights — there is no Adam warm-up phase.

## Code (`src/TensorFlow/`, despite the folder name — this part is pure JAX)

| Experiment | Script |
|---|---|
| Forward problem | `pinn_1d_chaleur_JAX_adam_normalized_comparaison_gpt_LBFGS_hard_BC_hard_IC.py` |
| Inverse problem, noise-free | `pinn_1d_chaleur_JAX_inverse_LBFGS_hard_BC_hard_IC_robustesse_pb_reel_dataset0.py` |
| Inverse problem, noisy data | `pinn_1d_chaleur_JAX_inverse_LBFGS_hard_BC_hard_IC_robustesse_pb_reel_dataset0_N_data_pts_bruit.py` |
| Data-loss weighting study | `pinn_1d_chaleur_JAX_inverse_LBFGS_hard_BC_hard_IC_robustesse_pb_reel_dataset0_N_data_pts_bruit_data_loss_mod.py` |

The first three scripts are self-contained. The weighting-study script relies on shared
modules also included here: `model.py`, `training.py`, `physique.py`, `plotting.py`,
`summary.py`, `utils.py`, `generate_noisy_dataset.py`, `drive_utils.py`.

The noise-free inverse script reads `heat1d_inverse_noisy_dataset.csv` (included) instead of
regenerating data on the fly; the other two generate their dataset at runtime.

### Known limitation

`drive_utils.py` creates a Windows `subst` drive (to work around Windows' path-length limit
for the long, parameter-derived result folder names). `BASE_PATH` in the weighting-study
script is resolved automatically from the script's own location, but the `subst` mechanism
itself is Windows-only and won't work on macOS/Linux.

### Dependencies

`jax`, `jaxopt`, `numpy`, `pandas`, `scipy`, `matplotlib`, `tqdm`, `openpyxl`.

## Report

`PINN_report/main.tex` (with `PINN_report/sections/` and `PINN_report/figures/`) is the
LaTeX source of the project report; `PINN_report/main.pdf` is the compiled version.
