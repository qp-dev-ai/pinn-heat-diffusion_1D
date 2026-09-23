#### 1D Heat equation PINN - comparaison 3 versions
# 1) non normalisé
# 2) normalisation physique seule
# 3) normalisation physique + centrage [-1,1]
#
# VERSION L-BFGS SEUL - REECRITURE JAX
# AVEC BC DURE + IC DURE
#
# Idée :
# - on impose exactement :
#       T(0,x) = sin(pi x / L)
#       T(t,0) = 0
#       T(t,L) = 0
# - le réseau n'apprend plus directement T
# - il apprend seulement un correctif multiplié par un facteur
#   qui s'annule sur l'IC et les BC
#
# Conséquence :
# - plus de loss IC
# - plus de loss BC
# - optimisation uniquement sur la PDE
#
# Ansatzs utilisés :
#
# 1) non normalisé (a=t, b=x):
#    T = sin(pi b / L) + a * b * (L-b) * N(a,b)
#
# 2) nd_only (a=tau, b=xi):
#    T = sin(pi b) + a * b * (1-b) * N(a,b)
#
# 3) nd_centered (a=t_net, b=x_net):
#    t=0  <=> a=-1
#    x=0  <=> b=-1
#    x=L  <=> b=+1
#    xi = (b+1)/2
#
#    T = sin(pi * (b+1)/2) + (a+1) * (1-b^2) * N(a,b)

import time
timing_INITIAL = time.time()
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import jax
import jax.numpy as jnp
from jax import grad, vmap
from jaxopt import LBFGS
from jax.experimental.compilation_cache import compilation_cache as cc

cc.set_cache_dir("./jax_cache") # PERMET DE GARDER EN CACHE LES COMPILATIONS JAX ENTRE LES RUNS, CE QUI ACCELERE BEAUCOUP LES EXÉCUTIONS SUIVANTES APRÈS LA PREMIÈRE.

mpl.rcParams["figure.max_open_warning"] = 100

# =========================================================
# CHOIX DU TYPE NUMERIQUE
# =========================================================
jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64

# =========================================================
# PARAMETRES GLOBAUX
# =========================================================
max_t = 5.0
L_barre = 1.0
alpha_val = 0.1

width_nn = 32
depth_nn = 3
f_activation = "tanh"

# Nombre de points fixes pour la PDE
N_f = 3000

# On garde ces paramètres pour monitoring / vérification seulement
N_ic = 100
N_bc = 100

# Avec hard IC + hard BC, la loss totale est PDE seule
lambda_ic = 0.0
lambda_bc = 0.0
lambda_phys = 1.0

# Grille de test
N_test = 50

# Nombre maximal d'itérations L-BFGS
N_epoch = 3000

# Fréquence d'enregistrement / affichage
snap_every = max(1, N_epoch // 10)
append_snap = 1
append_snap_common = max(1, snap_every // 10)

# Test final aléatoire
N_rand_final = 10000

pi = jnp.array(np.pi, dtype=DTYPE)
alpha = jnp.array(alpha_val, dtype=DTYPE)
L_tf = jnp.array(L_barre, dtype=DTYPE)
tmax_tf = jnp.array(max_t, dtype=DTYPE)

Fo = alpha_val * max_t / (L_barre ** 2)
Fo_tf = jnp.array(Fo, dtype=DTYPE)

print(f"temps max de simulation max_t = {max_t}")
print(f"longueur de la barre L = {L_barre}")
print(f"diffusivité thermique alpha = {alpha_val}")
print(f"Fourier number Fo = alpha * t_max / L^2 = {Fo:.6f}")

# =========================================================
# ACTIVATION
# =========================================================
def get_activation(name):
    if name == "tanh":
        return jnp.tanh
    elif name == "swish":
        return jax.nn.swish
    else:
        raise ValueError(f"Activation inconnue: {name}")

activation_fn = get_activation(f_activation)

# =========================================================
# MODELE BRUT
# =========================================================
def glorot_init(key, in_dim, out_dim, dtype=DTYPE):
    limit = np.sqrt(6.0 / (in_dim + out_dim))
    W = jax.random.uniform(
        key,
        shape=(in_dim, out_dim),
        minval=-limit,
        maxval=limit,
        dtype=dtype
    )
    b = jnp.zeros((out_dim,), dtype=dtype)
    return W, b

def make_model_params(width=32, depth=3, activation="tanh"):
    layer_dims = [2] + [width] * depth + [1]

    params = []
    for k in range(len(layer_dims) - 1):
        seed = 100 + k if k < depth else 200
        key = jax.random.PRNGKey(seed)
        W, b = glorot_init(key, layer_dims[k], layer_dims[k + 1], dtype=DTYPE)
        params.append({"W": W, "b": b})

    return params

def model_apply_raw(params, inputs):
    """
    Sortie brute du réseau sur un batch d'entrées de shape (N, 2).
    """
    x = inputs
    for layer in params[:-1]:
        x = x @ layer["W"] + layer["b"]
        x = activation_fn(x)

    x = x @ params[-1]["W"] + params[-1]["b"]
    return x

# =========================================================
# SOLUTION EXACTE
# =========================================================
def exact_solution(t_phys, x_phys):
    """
    Solution exacte :
    T(x,t) = exp(-alpha (pi/L)^2 t) sin(pi x / L)
    """
    T = jnp.exp(-alpha * ((pi / L_tf) ** 2) * t_phys) * jnp.sin((pi / L_tf) * x_phys)

    # Pour éviter l'effet numerique sin(pi) ~ 1e-16 aux bords
    T = jnp.where(jnp.isclose(x_phys, 0.0, atol=1e-14), 0.0, T)
    T = jnp.where(jnp.isclose(x_phys, L_tf, atol=1e-14), 0.0, T)
    return T

# =========================================================
# STRATEGIES DE NORMALISATION
# =========================================================
def phys_to_non_normalized(t_phys, x_phys):
    return t_phys, x_phys

def phys_to_nd_only(t_phys, x_phys):
    tau = alpha * t_phys / (L_tf ** 2)
    xi = x_phys / L_tf
    return tau, xi

def phys_to_nd_centered(t_phys, x_phys):
    xi = x_phys / L_tf
    tau = alpha * t_phys / (L_tf ** 2)

    x_net = 2.0 * xi - 1.0
    t_net = 2.0 * tau / Fo_tf - 1.0

    return t_net, x_net

def safe_ic_non_normalized(x):
    ic = jnp.sin((pi / L_tf) * x)
    ic = jnp.where(jnp.isclose(x, 0.0, atol=1e-14), 0.0, ic)
    ic = jnp.where(jnp.isclose(x, L_tf, atol=1e-14), 0.0, ic)
    return ic

def safe_ic_nd_only(xi):
    ic = jnp.sin(pi * xi)
    ic = jnp.where(jnp.isclose(xi, 0.0, atol=1e-14), 0.0, ic)
    ic = jnp.where(jnp.isclose(xi, 1.0, atol=1e-14), 0.0, ic)
    return ic

def safe_ic_centered(b):
    xi = 0.5 * (b + 1.0)
    ic = jnp.sin(pi * xi)
    ic = jnp.where(jnp.isclose(b, -1.0, atol=1e-14), 0.0, ic)
    ic = jnp.where(jnp.isclose(b,  1.0, atol=1e-14), 0.0, ic)
    return ic

# =========================================================
# HARD IC + HARD BC : ANSATZ CONTRAINT
# =========================================================
def make_model_apply_hard_ic_bc(mode):
    """
    Retourne une fonction model_apply(params, inputs) qui impose exactement :
        T(0,x) = sin(pi x / L)
        T(t,0) = 0
        T(t,L) = 0

    selon les coordonnées réseau (a,b).
    """
    def model_apply_constrained(params, inputs):
        a = inputs[:, 0:1]   # coordonnée temps réseau
        b = inputs[:, 1:2]   # coordonnée espace réseau
        T_raw = model_apply_raw(params, inputs)

        if mode == "non_normalized":
             ic_part = safe_ic_non_normalized(b)
             vanish_factor = a * b * (L_tf - b)

        elif mode == "nd_only":
             ic_part = safe_ic_nd_only(b)
             vanish_factor = a * b * (1.0 - b)

        elif mode == "nd_centered":
             ic_part = safe_ic_centered(b)
             vanish_factor = (a + 1.0) * (1.0 - b**2)

        else:
            raise ValueError("Mode inconnu")

        return ic_part + vanish_factor * T_raw

    return model_apply_constrained

# =========================================================
# RESIDUS PDE SELON LA STRATEGIE
# =========================================================
def scalar_model_output(params, a, b, model_apply):
    """
    Sortie scalaire du modèle contraint sur un seul point (a, b).
    """
    inp = jnp.array([[a, b]], dtype=DTYPE)
    return model_apply(params, inp)[0, 0]

def make_physics_residual(mode, model_apply):
    """
    Résidu natif PDE calculé sur la sortie CONTRAINTE.
    """
    def residual_single(params, a, b):
        dT_da = grad(scalar_model_output, argnums=1)(params, a, b, model_apply)
        d2T_db2 = grad(grad(scalar_model_output, argnums=2), argnums=2)(params, a, b, model_apply)

        if mode == "non_normalized":
            # a=t, b=x
            r = dT_da - alpha * d2T_db2

        elif mode == "nd_only":
            # a=tau, b=xi
            r = dT_da - d2T_db2

        elif mode == "nd_centered":
            # a=t_net, b=x_net
            r = dT_da - 2.0 * Fo_tf * d2T_db2

        else:
            raise ValueError("Mode inconnu")

        return r

    residual_batch = vmap(residual_single, in_axes=(None, 0, 0))

    def physics_residual(params, tx):
        a = tx[:, 0]
        b = tx[:, 1]
        r = residual_batch(params, a, b)
        return r.reshape(-1, 1)

    return physics_residual

def residual_native_to_physical(r_native, mode):
    """
    Convertit le résidu natif vers le résidu physique :
        T_t - alpha T_xx
    """
    if mode == "non_normalized":
        return r_native
    elif mode == "nd_only":
        return (alpha / (L_tf ** 2)) * r_native
    elif mode == "nd_centered":
        return (2.0 / tmax_tf) * r_native
    else:
        raise ValueError("Mode inconnu")

# =========================================================
# OUTILS
# =========================================================
def mse(a, b):
    return jnp.mean((a - b) ** 2)

# =========================================================
# EXPERIENCE L-BFGS
# =========================================================
def run_experiment(mode, title, base_params):
    print("\n" + "=" * 70)
    print(f"EXPERIENCE: {title}")
    print("=" * 70)

    np.random.seed(0)

    params = copy.deepcopy(base_params)

    if mode == "non_normalized":
        phys_to_net = phys_to_non_normalized
    elif mode == "nd_only":
        phys_to_net = phys_to_nd_only
    elif mode == "nd_centered":
        phys_to_net = phys_to_nd_centered
    else:
        raise ValueError("Mode inconnu")

    # Modèle contraint hard IC + hard BC
    model_apply = make_model_apply_hard_ic_bc(mode)

    # Résidu PDE calculé sur ce modèle contraint
    physics_residual = make_physics_residual(mode, model_apply)

    # =====================================================
    # POINTS IC (verification seulement)
    # =====================================================
    x_ic_phys = np.linspace(0.0, L_barre, N_ic).reshape(-1, 1).astype(np.float64)
    t_ic_phys = np.zeros_like(x_ic_phys).astype(np.float64)
    T_ic_np = np.sin(np.pi * x_ic_phys / L_barre).astype(np.float64)

    t_ic_net, x_ic_net = phys_to_net(
        jnp.array(t_ic_phys, dtype=DTYPE),
        jnp.array(x_ic_phys, dtype=DTYPE),
    )
    tx_ic_net = jnp.concatenate([t_ic_net, x_ic_net], axis=1)
    T_ic = jnp.array(T_ic_np, dtype=DTYPE)

    # =====================================================
    # POINTS BC (verification seulement)
    # =====================================================
    t_bc_phys = np.linspace(0.0, max_t, N_bc).reshape(-1, 1).astype(np.float64)
    x_bc0_phys = np.zeros_like(t_bc_phys).astype(np.float64)
    x_bc1_phys = (L_barre * np.ones_like(t_bc_phys)).astype(np.float64)

    t_bc0_net, x_bc0_net = phys_to_net(
        jnp.array(t_bc_phys, dtype=DTYPE),
        jnp.array(x_bc0_phys, dtype=DTYPE),
    )
    t_bc1_net, x_bc1_net = phys_to_net(
        jnp.array(t_bc_phys, dtype=DTYPE),
        jnp.array(x_bc1_phys, dtype=DTYPE),
    )

    tx_bc0_net = jnp.concatenate([t_bc0_net, x_bc0_net], axis=1)
    tx_bc1_net = jnp.concatenate([t_bc1_net, x_bc1_net], axis=1)

    # =====================================================
    # PDE FIXE
    # =====================================================
    t_f_phys = np.random.uniform(0.0, max_t, size=(N_f, 1)).astype(np.float64)
    x_f_phys = np.random.uniform(0.0, L_barre, size=(N_f, 1)).astype(np.float64)

    t_f_net, x_f_net = phys_to_net(
        jnp.array(t_f_phys, dtype=DTYPE),
        jnp.array(x_f_phys, dtype=DTYPE)
    )
    tx_f_net = jnp.concatenate([t_f_net, x_f_net], axis=1)

    # =====================================================
    # GRILLE DE TEST FIXE
    # =====================================================
    t_test_phys = jnp.linspace(jnp.array(0.0, dtype=DTYPE), jnp.array(max_t, dtype=DTYPE), N_test)
    x_test_phys = jnp.linspace(jnp.array(0.0, dtype=DTYPE), jnp.array(L_barre, dtype=DTYPE), N_test)

    Tg_phys, Xg_phys = jnp.meshgrid(t_test_phys, x_test_phys, indexing="ij")

    t_test_net, x_test_net = phys_to_net(
        Tg_phys.reshape(-1, 1),
        Xg_phys.reshape(-1, 1)
    )
    tx_test_net = jnp.concatenate([t_test_net, x_test_net], axis=1)

    T_true = exact_solution(Tg_phys, Xg_phys)

    data = {
        "tx_ic_net": tx_ic_net,
        "T_ic": T_ic,
        "tx_bc0_net": tx_bc0_net,
        "tx_bc1_net": tx_bc1_net,
        "tx_f_net": tx_f_net,
        "tx_test_net": tx_test_net,
        "T_true": T_true,
    }

    # =====================================================
    # LOSSES FIXES
    # =====================================================
    def compute_losses_fixed(params, data):
        """
        Hard IC + hard BC => optimisation sur la PDE uniquement.
        IC et BC sont juste monitorées pour vérifier que c'est bien exact.
        """
        # IC check
        T_ic_pred = model_apply(params, data["tx_ic_net"])
        loss_ic_check = mse(T_ic_pred, data["T_ic"])

        # BC check
        T_bc0_pred = model_apply(params, data["tx_bc0_net"])
        T_bc1_pred = model_apply(params, data["tx_bc1_net"])
        loss_bc_check = jnp.mean(T_bc0_pred ** 2) + jnp.mean(T_bc1_pred ** 2)

        # PDE
        r = physics_residual(params, data["tx_f_net"])
        loss_phys = jnp.mean(r ** 2)

        # totale
        loss = lambda_phys * loss_phys

        aux = {
            "loss_ic": loss_ic_check,
            "loss_bc": loss_bc_check,
            "loss_phys": loss_phys,
        }
        return loss, aux

    def loss_only(params, data):
        loss, _ = compute_losses_fixed(params, data)
        return loss

    compute_losses_fixed_jit = jax.jit(compute_losses_fixed)
    loss_only_jit = jax.jit(loss_only)

    # =====================================================
    # EVALUATION COMMUNE SUR GRILLE FIXE
    # =====================================================
    def evaluate_common_metrics(params, data):
        loss, aux = compute_losses_fixed(params, data)

        T_pred_flat_common = model_apply(params, data["tx_test_net"])
        T_pred_common = T_pred_flat_common.reshape(N_test, N_test)
        mse_solution = mse(T_pred_common, data["T_true"])

        r_native_grid = physics_residual(params, data["tx_test_net"])
        r_phys_grid = residual_native_to_physical(r_native_grid, mode)
        loss_phys_common = jnp.mean(r_phys_grid ** 2)

        loss_total_common = loss_phys_common

        return {
            "loss": loss,
            "loss_ic": aux["loss_ic"],
            "loss_bc": aux["loss_bc"],
            "loss_phys": aux["loss_phys"],
            "loss_phys_common": loss_phys_common,
            "loss_total_common": loss_total_common,
            "mse_solution": mse_solution,
            "T_pred_common": T_pred_common,
        }

    evaluate_common_metrics_jit = jax.jit(evaluate_common_metrics)

    # =====================================================
    # PREPARATION L-BFGS
    # =====================================================
    epoch_history = []
    epoch_common_history = []

    loss_history = []
    loss_ic_history = []
    loss_bc_history = []
    loss_phys_history = []

    loss_phys_common_history = []
    loss_total_common_history = []
    mse_solution_history = []

    T_snaps = []
    epoch_snaps = []

    solver = LBFGS(
        fun=loss_only_jit,
        maxiter=N_epoch,
        tol=1e-9,
        history_size=10,
        linesearch="zoom",
        jit=True,
    )

    state = solver.init_state(params, data)

    # =====================================================
    # LANCEMENT L-BFGS
    # =====================================================
    timing_before_train = time.time()

    converged = False
    failed = False
    final_loss = None
    num_iterations = 0

    for k in range(1, N_epoch + 1):
        params, state = solver.update(params, state, data)
        num_iterations = k

        if k % append_snap == 0:
            loss_val, aux = compute_losses_fixed_jit(params, data)

            epoch_history.append(k)
            loss_history.append(float(loss_val))
            loss_ic_history.append(float(aux["loss_ic"]))
            loss_bc_history.append(float(aux["loss_bc"]))
            loss_phys_history.append(float(aux["loss_phys"]))

        if k % append_snap_common == 0:
            metrics = evaluate_common_metrics_jit(params, data)

            epoch_common_history.append(k)
            mse_solution_history.append(float(metrics["mse_solution"]))
            loss_phys_common_history.append(float(metrics["loss_phys_common"]))
            loss_total_common_history.append(float(metrics["loss_total_common"]))

        if k % snap_every == 0:
            metrics = evaluate_common_metrics_jit(params, data)

            T_snaps.append(np.array(metrics["T_pred_common"]))
            epoch_snaps.append(k)

            print(
                f"it {k} | "
                f"total loss (PDE fixe): {float(metrics['loss']):.3e} | "
                f"IC check: {float(metrics['loss_ic']):.3e} | "
                f"BC check: {float(metrics['loss_bc']):.3e} | "
                f"loss phys (native, fixe): {float(metrics['loss_phys']):.3e} | "
                f"MSE solution grille: {float(metrics['mse_solution']):.3e} | "
                f"loss phys (commune, grille): {float(metrics['loss_phys_common']):.3e}"
            )

        if np.isnan(float(state.value)) or np.isinf(float(state.value)):
            failed = True
            final_loss = float(state.value)
            print("Arrêt : NaN / Inf détecté dans l'objectif.")
            break

        if float(state.error) < 1e-9:
            converged = True
            final_loss = float(state.value)
            break

    if final_loss is None:
        final_loss = float(state.value)
        converged = bool(float(state.error) < 1e-9)

    total_time = time.time() - timing_before_train
    print(f"Temps total: {total_time:.1f}s")
    print(f"Converged: {converged}")
    print(f"Failed   : {failed}")
    print(f"Iterations L-BFGS: {num_iterations}")
    print(f"Final objective: {final_loss:.3e}")

    # =====================================================
    # TEST FINAL ALEATOIRE
    # =====================================================
    np.random.seed(0)
    t_final_phys = np.random.uniform(0.0, max_t, size=(N_rand_final, 1)).astype(np.float64)
    x_final_phys = np.random.uniform(0.0, L_barre, size=(N_rand_final, 1)).astype(np.float64)

    t_final_phys = jnp.array(t_final_phys, dtype=DTYPE)
    x_final_phys = jnp.array(x_final_phys, dtype=DTYPE)

    t_final_net, x_final_net = phys_to_net(t_final_phys, x_final_phys)
    tx_final_net = jnp.concatenate([t_final_net, x_final_net], axis=1)

    T_pred_final = model_apply(params, tx_final_net)
    T_true_final = exact_solution(t_final_phys, x_final_phys)

    mse_final = jnp.mean((T_pred_final - T_true_final) ** 2)

    r_test = physics_residual(params, tx_final_net)
    mse_residual_native = jnp.mean(r_test ** 2)
    r_phys = residual_native_to_physical(r_test, mode)
    mse_residual_phys = jnp.mean(r_phys ** 2)

    # Vérifications IC / BC finales
    T_ic_final = model_apply(params, data["tx_ic_net"])
    mse_ic_final = jnp.mean((T_ic_final - data["T_ic"]) ** 2)

    T_bc0_final = model_apply(params, data["tx_bc0_net"])
    T_bc1_final = model_apply(params, data["tx_bc1_net"])
    mse_bc_final = jnp.mean(T_bc0_final ** 2) + jnp.mean(T_bc1_final ** 2)

    print(f"MSE solution finale (pts aléatoires)       : {float(mse_final):.3e}")
    print(f"MSE résidu native finale (pts aléatoires)  : {float(mse_residual_native):.3e}")
    print(f"MSE résidu physique final (pts aléatoires) : {float(mse_residual_phys):.3e}")
    print(f"MSE IC finale (check)                      : {float(mse_ic_final):.3e}")
    print(f"MSE BC finale (check)                      : {float(mse_bc_final):.3e}")

    if len(T_snaps) == 0:
        metrics = evaluate_common_metrics_jit(params, data)
        T_snaps.append(np.array(metrics["T_pred_common"]))
        epoch_snaps.append(num_iterations)

    return {
        "title": title,
        "mode": mode,
        "params": params,
        "epoch_history": epoch_history,
        "epoch_common_history": epoch_common_history,
        "loss_history": loss_history,
        "loss_ic_history": loss_ic_history,
        "loss_bc_history": loss_bc_history,
        "loss_phys_history": loss_phys_history,
        "T_snaps": T_snaps,
        "epoch_snaps": epoch_snaps,
        "T_true": np.array(T_true),
        "t_test_phys": np.array(t_test_phys),
        "x_test_phys": np.array(x_test_phys),
        "t_final_phys": np.array(t_final_phys),
        "x_final_phys": np.array(x_final_phys),
        "loss_phys_common_history": loss_phys_common_history,
        "loss_total_common_history": loss_total_common_history,
        "mse_solution_history": mse_solution_history,
        "mse_final": float(mse_final),
        "mse_residual_native": float(mse_residual_native),
        "mse_residual_phys": float(mse_residual_phys),
        "mse_ic_final": float(mse_ic_final),
        "mse_bc_final": float(mse_bc_final),
        "num_iterations": int(num_iterations),
        "objective_value": float(final_loss),
        "converged": bool(converged),
        "failed": bool(failed),
    }

# =========================================================
# POIDS INITIAUX COMMUNS
# =========================================================
base_params = make_model_params(width_nn, depth_nn, f_activation)

# =========================================================
# RUN DES 3 EXPERIENCES
# =========================================================
results_non = run_experiment("non_normalized", "1) Non normalisé", base_params)
results_nd  = run_experiment("nd_only", "2) Nondimensionnalisation physique seule", base_params)
results_ndc = run_experiment("nd_centered", "3) Nondimensionnalisation physique + centrage [-1,1]", base_params)

all_results = [results_non, results_nd, results_ndc]

# =========================================================
# RESUME NUMERIQUE
# =========================================================
print("\n" + "=" * 70)
print("RESUME FINAL (points aléatoires)")
print("=" * 70)
for res in all_results:
    print(
        f"{res['title']:<45} | "
        f"MSE sol (aléatoire) = {res['mse_final']:.3e} | "
        f"MSE residu phys (aléatoire) = {res['mse_residual_phys']:.3e} | "
        f"MSE IC check = {res['mse_ic_final']:.3e} | "
        f"MSE BC check = {res['mse_bc_final']:.3e} | "
        f"iters L-BFGS = {res['num_iterations']}"
    )

total_FINAL_time = time.time() - timing_INITIAL
print(f"Temps FINAL total: {total_FINAL_time:.1f}s")

# =========================================================
# PLOT LOSSES COMPARATIVES
# =========================================================
plt.figure()
for res in all_results:
    plt.plot(res["epoch_history"], res["loss_history"], label=res["title"])
plt.xlabel("Itérations / appels L-BFGS")
plt.ylabel("PDE Loss")
plt.title("Comparaison des total losses (hard IC + hard BC)")
plt.legend()

plt.figure()
for res in all_results:
    plt.plot(res["epoch_common_history"], res["loss_total_common_history"], label=res["title"])
plt.xlabel("Itérations / appels L-BFGS")
plt.ylabel("Physics Loss (physical, fixed grid)")
plt.title("Comparaison des physics losses physiques sur grille fixe")
plt.legend()

plt.figure()
for res in all_results:
    plt.plot(res["epoch_history"], res["loss_phys_history"], label=res["title"])
plt.xlabel("Itérations / appels L-BFGS")
plt.ylabel("Physics Loss")
plt.title("Comparaison des physics losses (native, points fixes)")
plt.legend()

plt.figure()
for res in all_results:
    plt.plot(res["epoch_history"], res["loss_ic_history"], label=res["title"])
plt.xlabel("Itérations / appels L-BFGS")
plt.ylabel("IC check")
plt.title("Vérification IC dure (doit rester ~ 0)")
plt.legend()

plt.figure()
for res in all_results:
    plt.plot(res["epoch_history"], res["loss_bc_history"], label=res["title"])
plt.xlabel("Itérations / appels L-BFGS")
plt.ylabel("BC check")
plt.title("Vérification BC dure (doit rester ~ 0)")
plt.legend()

plt.figure()
for res in all_results:
    plt.plot(res["epoch_common_history"], res["mse_solution_history"], label=res["title"])
plt.xlabel("Itérations / appels L-BFGS")
plt.ylabel("MSE solution on fixed grid")
plt.title("Comparaison des MSE solution sur grille fixe")
plt.legend()

# =========================================================
# SNAPSHOTS COMPARATIFS FINAUX
# =========================================================
x_vals = np.linspace(0.0, L_barre, 6)
t_vals = np.linspace(0.0, max_t, 6)

for res in all_results:
    T_last = res["T_snaps"][-1]
    T_true = res["T_true"]
    t_np = res["t_test_phys"]
    x_np = res["x_test_phys"]

    j_list = [int(round((xv / L_barre) * (N_test - 1))) for xv in x_vals]
    i_list = [int(round((tv / max_t) * (N_test - 1))) for tv in t_vals]

    plt.figure()
    for j, xv in zip(j_list, x_vals):
        line_model, = plt.plot(t_np, T_last[:, j], label=f"modèle | x={xv:.2f}")
        color = line_model.get_color()
        plt.plot(t_np, T_true[:, j], "--", color=color, label=f"exact | x={xv:.2f}" if j == j_list[0] else None)
    plt.xlabel("t")
    plt.ylabel("T")
    plt.title(f"T(t) final - {res['title']}")
    plt.legend(fontsize=8)

    plt.figure()
    for i, tv in zip(i_list, t_vals):
        line_model, = plt.plot(x_np, T_last[i, :], label=f"modèle | t={tv:.2f}")
        color = line_model.get_color()
        plt.plot(x_np, T_true[i, :], "--", color=color, label=f"exact | t={tv:.2f}" if i == i_list[0] else None)
    plt.xlabel("x")
    plt.ylabel("T")
    plt.title(f"T(x) final - {res['title']}")
    plt.legend(fontsize=8)

# =========================================================
# COMPARAISON DIRECTE DES 3 MODELES SUR UN MEME GRAPHE
# =========================================================
x_compare = [0.0, 0.5 * L_barre, L_barre]
j_compare = [int(round((xv / L_barre) * (N_test - 1))) for xv in x_compare]

for j, xv in zip(j_compare, x_compare):
    plt.figure()
    for res in all_results:
        T_last = res["T_snaps"][-1]
        plt.plot(res["t_test_phys"], T_last[:, j], label=res["title"])
    plt.plot(results_non["t_test_phys"], results_non["T_true"][:, j], "k--", label="Exact")
    plt.xlabel("t")
    plt.ylabel("T")
    plt.title(f"Comparaison T(t) à x={xv:.2f}")
    plt.legend()

t_compare = [0.0, 0.5 * max_t, max_t]
i_compare = [int(round((tv / max_t) * (N_test - 1))) for tv in t_compare]

for i, tv in zip(i_compare, t_compare):
    plt.figure()
    for res in all_results:
        T_last = res["T_snaps"][-1]
        plt.plot(res["x_test_phys"], T_last[i, :], label=res["title"])
    plt.plot(results_non["x_test_phys"], results_non["T_true"][i, :], "k--", label="Exact")
    plt.xlabel("x")
    plt.ylabel("T")
    plt.title(f"Comparaison T(x) à t={tv:.2f}")
    plt.legend()

plt.show()
plt.close("all")

