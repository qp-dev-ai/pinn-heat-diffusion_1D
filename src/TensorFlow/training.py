import time
import copy # Pour les copy profondes
from collections import deque # Pour le critère d'arrêt basé sur les 100 dernières itérations, on utilise une deque pour stocker les valeurs de MSE solution, MSE physique, MSE data, MSE IC et MSE BC des 100 dernières itérations. En utilisant deque avec maxlen=WINDOW_STOP, on s'assure que la deque ne stocke que les 100 dernières valeurs, ce qui permet de calculer facilement la moyenne de ces valeurs pour déterminer si le critère d'arrêt est satisfait. Cela permet d'arrêter l'entraînement lorsque la performance du modèle ne s'améliore plus de manière significative sur ces métriques pendant un certain nombre d'itérations, ce qui peut aider à éviter un surentraînement et à économiser du temps de calcul.

import numpy as np
import jax
import jax.numpy as jnp
from jaxopt import LBFGS 
from jax.flatten_util import ravel_pytree # Pour aplatir les paramètres du modèle en un vecteur 1D, ce qui est nécessaire pour utiliser l'optimiseur L-BFGS de jaxopt. En utilisant ravel_pytree, on peut convertir les paramètres du modèle, qui sont généralement organisés en structures de données complexes (comme des listes ou des dictionnaires de matrices de poids et de biais), en un vecteur 1D qui peut être facilement manipulé par l'optimiseur L-BFGS. Cela permet d'optimiser efficacement les paramètres du modèle en utilisant L-BFGS tout en conservant la structure originale des paramètres pour les calculs de la perte et des gradients pendant l'entraînement.
from scipy.stats import qmc # Pour le Latin Hypercube Sampling (LHS) pour la génération de points de la PDE
from tqdm.auto import tqdm # Pour afficher une barre de progression pendant l'entraînement, ce qui permet de suivre l'avancement de l'entraînement en temps réel. En utilisant tqdm, on peut facilement visualiser le nombre d'itérations effectuées, le temps restant estimé et d'autres informations utiles pendant l'entraînement du modèle, ce qui peut aider à mieux comprendre le processus d'entraînement et à identifier les moments où le modèle commence à converger ou à rencontrer des difficultés pour satisfaire la PDE.

from utils import clean_small, format_time
from model import get_alpha
from physique import (
    phys_to_non_normalized,
    exact_solution,
    make_model_apply_hard_ic_bc,
    make_physics_residual,
    residual_native_to_physical,
    mse,
)
from plotting import save_run_plots

# =========================================================
# EXPERIENCE L-BFGS
# =========================================================
def run_experiment(
    mode,
    title,
    base_train_params,
    lambda_data_var,
    seed_var,
    n_coloc,
    run_dir,
    run_idx,
    total_runs,
    data_arrays,
    config,
    model_apply_raw,
    activation_fn,
):
    print("\n" + "=" * 70)
    progress_pct = 100.0 * run_idx / total_runs
    print(f"EXPERIENCE {run_idx}/{total_runs} ({progress_pct:.1f}%) : {title}")
    print("=" * 70)

    # np.random.seed(0) # plus utile maintenant car seed deja dans le LHS

    train_params = copy.deepcopy(base_train_params)

    DTYPE = config["DTYPE"]
    L_barre = config["L_barre"]
    max_t = config["max_t"]
    L_tf = config["L_tf"]
    alpha_theo = config["alpha_theo"]
    alpha_fit = config["alpha_fit"]
    N_ic = config["N_ic"]
    N_bc = config["N_bc"]
    N_test = config["N_test"]
    N_epoch = config["N_epoch"]
    N_rand_final = config["N_rand_final"]
    snap_every = config["snap_every"]
    append_snap = config["append_snap"]
    append_snap_common = config["append_snap_common"]
    WINDOW_STOP = config["WINDOW_STOP"]
    THRESH_MSE_SOLUTION = config["THRESH_MSE_SOLUTION"]
    THRESH_MSE_PHYS = config["THRESH_MSE_PHYS"]
    THRESH_MSE_DATA = config["THRESH_MSE_DATA"]
    THRESH_MSE_IC = config["THRESH_MSE_IC"]
    THRESH_MSE_BC = config["THRESH_MSE_BC"]
    lambda_phys = config["lambda_phys"]
    T_threshold = config["T_threshold"]
    window_noise = config["window_noise"]
    method_noise = config["method_noise"]
    crit_noise = config["crit_noise"]
    noise_estimation = config["noise_estimation"]

    x_data_jax = data_arrays["x_data_jax"]
    t_data_jax = data_arrays["t_data_jax"]
    T_obs_jax = data_arrays["T_obs_jax"]
    poids_alpha_info_jax = data_arrays["poids_alpha_info_jax"]

    if mode == "non_normalized":
        phys_to_net = phys_to_non_normalized
    # elif mode == "nd_only":
    #     phys_to_net = phys_to_nd_only
    # elif mode == "nd_centered":
    #     phys_to_net = phys_to_nd_centered
    else:
        raise ValueError("Mode inconnu")

    # Modèle contraint hard IC + hard BC
    model_apply = make_model_apply_hard_ic_bc(
    mode,
    model_apply_raw,
    activation_fn,
    L_tf,
)

    # Résidu PDE calculé sur ce modèle contraint
    physics_residual = make_physics_residual(  #  On calcule le résidu apres avoir calculé, pour que le résidu soit calculé sur la sortie contrainte du modèle, ce qui garantit que les conditions initiales et aux bords sont respectées de manière stricte pendant l'entraînement. En utilisant physics_residual, on peut ensuite calculer la perte de la PDE en utilisant ce résidu, ce qui guide l'optimisation du modèle pour mieux satisfaire la PDE tout en respectant les contraintes imposées par les conditions initiales et aux bords.
    mode,
    model_apply,
    get_alpha,
) 

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
    # PDE FIXE (POINTS DE COLOC LATIN HYPERCUBE SAMPLING pour la PDE, pour que les points soient les mêmes à chaque run et pour que ce soit plus efficace que du random sampling) : on génère N_f points de la PDE en utilisant un échantillonnage de Latin Hypercube (LHS) pour garantir une distribution uniforme des points dans l'espace de la PDE. En utilisant qmc.LatinHypercube, on peut générer des points de la PDE qui couvrent efficacement l'espace de la PDE, ce qui permet d'obtenir une meilleure approximation de la solution et d'améliorer la performance du modèle pendant l'entraînement. Les points générés sont ensuite convertis en coordonnées réseau à l'aide de phys_to_net pour être utilisés dans le calcul du résidu de la PDE pendant l'entraînement du modèle.
    # =====================================================
    sampler = qmc.LatinHypercube(d=2, seed=seed_var, optimization="random-cd") # d=2 pour 2 dimensions (t,x), seed=0 pour garantir que les points générés sont les mêmes à chaque exécution du code, ce qui permet une comparaison équitable entre les différentes stratégies d'entraînement et d'optimisation utilisées dans les expériences, optimization="random-cd" pour améliorer la qualité de l'échantillonnage en utilisant une stratégie de "centered" randomisée, ce qui peut aider à obtenir une meilleure couverture de l'espace de la PDE et ainsi améliorer la performance du modèle pendant l'entraînement.
    sample = sampler.random(n=n_coloc)
    t_f_phys = sample[:, 0:1] * max_t
    x_f_phys = sample[:, 1:2] * L_barre

    t_f_net, x_f_net = phys_to_net( # Suivant le mode (normalisé ou non), on convertit les coordonnées physiques des points de la PDE en coordonnées réseau, ce qui permet de calculer le résidu de la PDE sur ces points pendant l'entraînement du modèle. En utilisant phys_to_net, on s'assure que les points utilisés pour calculer la perte de la PDE sont correctement transformés en fonction de la stratégie de normalisation choisie, ce qui est important pour garantir que le modèle puisse apprendre à satisfaire la PDE de manière précise et efficace.
        jnp.array(t_f_phys, dtype=DTYPE),
        jnp.array(x_f_phys, dtype=DTYPE)
    )
    tx_f_net = jnp.concatenate([t_f_net, x_f_net], axis=1)

    ### Conversion des points de DATA en fonction de la normalisation choisie, pour que les points de données soient cohérents avec les points de la PDE pendant l'entraînement du modèle. En utilisant phys_to_net, on peut convertir les coordonnées physiques des points de données en coordonnées réseau, ce qui permet de calculer la perte de données en comparant les prédictions du modèle avec les observations pour ces points pendant l'entraînement. Cela garantit que le modèle apprend à faire des prédictions précises pour les points de données observées tout en respectant la stratégie de normalisation choisie, ce qui peut aider à améliorer la précision globale du modèle et sa capacité à satisfaire la PDE.
    t_data_net, x_data_net = phys_to_net(t_data_jax, x_data_jax)
    tx_data_net = jnp.concatenate([t_data_net, x_data_net], axis=1)


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

    T_true = exact_solution(Tg_phys, Xg_phys, alpha_theo, L_tf) # Calcul de la solution exacte sur la grille de test, qui est utilisée pour évaluer la performance du modèle en comparant les prédictions du modèle avec cette solution exacte. En utilisant T_true, on peut calculer des métriques d'évaluation telles que le MSE entre les prédictions du modèle et la solution exacte, ce qui permet de mesurer la précision du modèle et sa capacité à apprendre à satisfaire la PDE de manière précise.

    data = {
        "tx_ic_net": tx_ic_net,
        "T_ic": T_ic,
        "tx_bc0_net": tx_bc0_net,
        "tx_bc1_net": tx_bc1_net,
        "tx_data_net": tx_data_net,
        "tx_f_net": tx_f_net,
        "tx_test_net": tx_test_net,
        "T_true": T_true,
    }

    # =====================================================
    # LOSSES FIXES
    # =====================================================
    def compute_losses_fixed(train_params, data):
        """
        Hard IC + hard BC => optimisation sur la PDE + DATA uniquement.
        IC et BC sont juste monitorées pour vérifier que c'est bien exact.
        """
        # IC check
        T_ic_pred = model_apply(train_params["nn"], data["tx_ic_net"]) # model apply ne prends que les poids du réseaux donc train_params["nn"] important
        loss_ic_check = mse(T_ic_pred, data["T_ic"])

        # BC check
        T_bc0_pred = model_apply(train_params["nn"], data["tx_bc0_net"]) # model apply ne prends que les poids du réseaux donc train_params["nn"] important
        T_bc1_pred = model_apply(train_params["nn"], data["tx_bc1_net"])
        loss_bc_check = jnp.mean(T_bc0_pred ** 2) + jnp.mean(T_bc1_pred ** 2)

        # DATA
        T_pred_obs = model_apply(train_params["nn"], data["tx_data_net"])
        err2_data = (T_pred_obs - T_obs_jax) ** 2

        if config["loss_mode"] == "data_loss_non_mod":
            loss_data = jnp.mean(err2_data)

        elif config["loss_mode"] == "data_loss_mod_signal_only":
            poids_data_loss = T_obs_jax**2 / (T_obs_jax**2 + T_threshold**2)
            loss_data = jnp.sum(poids_data_loss * err2_data) / (
                jnp.sum(poids_data_loss) + 1e-12
            )

        elif config["loss_mode"] == "data_loss_mod_alpha_info": # On multiplie chaque terme de la loss data par le poids d'information de alpha pour ce point, pour que les points où alpha est plus informatif aient un poids plus important dans la loss data, ce qui peut aider à améliorer la précision du modèle pour ces points et ainsi mieux satisfaire la PDE.
            loss_data = jnp.sum(poids_alpha_info_jax * err2_data) / (
                jnp.sum(poids_alpha_info_jax) + 1e-12
            )

        else:
            raise ValueError(f"loss_mode inconnu: {config['loss_mode']}")
        
        # PDE
        r = physics_residual(train_params, data["tx_f_net"]) # Ici par contre prend en compte alpha aussi
        loss_phys = jnp.mean(r ** 2)

                # totale
        loss = lambda_data_var*loss_data + lambda_phys*loss_phys 

        aux = {
            "loss_ic": loss_ic_check,
            "loss_bc": loss_bc_check,
            "loss_data": loss_data,
            "loss_phys": loss_phys,
        }
        return loss, aux

    def loss_only(train_params, data): # Retourne la loss seul sans les auxiliaires, pour que le solver LBFGS puisse l'optimiser directement. En utilisant loss_only, on peut passer cette fonction au solver LBFGS pour qu'il puisse minimiser la perte totale pendant l'entraînement du modèle, en ajustant les paramètres du modèle pour réduire cette perte et ainsi améliorer la performance du modèle pour satisfaire la PDE et faire des prédictions précises.
        loss, _ = compute_losses_fixed(train_params, data)
        return loss

    compute_losses_fixed_jit = jax.jit(compute_losses_fixed) # IMPORTANT. JIT de la fonction compute_losses_fixed pour accélérer les calculs pendant l'entraînement du modèle, en compilant cette fonction avec JAX pour qu'elle puisse être exécutée de manière plus efficace sur le matériel disponible (CPU ou GPU). En utilisant compute_losses_fixed_jit, on peut calculer la perte et les métriques associées plus rapidement pendant l'entraînement, ce qui permet d'obtenir des résultats plus rapidement et d'améliorer l'efficacité globale de l'entraînement du modèle.
    loss_only_jit = jax.jit(loss_only)

    def grad_norm(train_params, data):
        grads = jax.grad(loss_only)(train_params, data)
        flat_grads, _ = ravel_pytree(grads)
        return jnp.linalg.norm(flat_grads)
    
    grad_norm_jit = jax.jit(grad_norm)

    # =====================================================
    # EVALUATION COMMUNE SUR GRILLE FIXE
    # =====================================================
    def evaluate_common_metrics(train_params, data):
        loss, aux = compute_losses_fixed(train_params, data)
        
        # Prediction solution sur grille de test
        T_pred_flat_common = model_apply(train_params["nn"], data["tx_test_net"])
        T_pred_common = T_pred_flat_common.reshape(N_test, N_test)
        mse_solution = mse(T_pred_common, data["T_true"])

        # Résidu PDE sur grille de test
        r_native_grid = physics_residual(train_params, data["tx_test_net"])
        r_phys_grid = residual_native_to_physical(r_native_grid, mode)
        loss_phys_common = jnp.mean(r_phys_grid ** 2)

        # Data loss
        T_pred_obs_common = model_apply(train_params["nn"], data["tx_data_net"])
        loss_data_common = jnp.mean((T_pred_obs_common - T_obs_jax)**2)

        # TOTAL LOSS COMMUNE

        loss_total_common = lambda_phys * loss_phys_common + lambda_data_var * loss_data_common 

        return {
            "loss": loss,
            "loss_ic": aux["loss_ic"],
            "loss_bc": aux["loss_bc"],
            "loss_phys": aux["loss_phys"],
            "loss_data": loss_data_common,
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
    loss_data_history = []
    loss_phys_common_history = []
    loss_total_common_history = []
    mse_solution_history = []

    alpha_val_history = []

    T_snaps = []
    epoch_snaps = []

    solver = LBFGS(
        fun=loss_only_jit, # Fonction de perte à minimiser par le solver LBFGS, qui prend en entrée les paramètres du modèle train_params et les données data, et retourne la valeur de la perte totale calculée par la fonction loss_only_jit. En utilisant loss_only_jit, on permet au solver LBFGS de minimiser directement la perte totale pendant l'entraînement du modèle, en ajustant les paramètres du modèle pour réduire cette perte et ainsi améliorer la performance du modèle pour satisfaire la PDE et faire des prédictions précises.
        maxiter=N_epoch, # Nombre maximal d'itérations pour le solver LBFGS, qui détermine combien de fois le solver effectuera une mise à jour des paramètres du modèle pendant l'entraînement. En utilisant maxiter=N_epoch, on fixe le nombre d'itérations à N_epoch, ce qui permet de contrôler la durée de l'entraînement et d'assurer que le solver s'arrête après un nombre raisonnable d'itérations, même si la convergence n'est pas atteinte. Cela peut être important pour éviter des temps d'entraînement excessifs dans les cas où la convergence est difficile à atteindre ou lorsque les ressources de calcul sont limitées.
        tol=1e-9, # Tolérance de convergence pour le solver LBFGS (defaut -9), qui détermine le critère d'arrêt de l'optimisation. En utilisant tol=1e-9, on impose que le solver s'arrête lorsque la norme du gradient de la fonction de perte est inférieure à 1e-9, ce qui indique que le modèle a atteint un point où les améliorations supplémentaires sont très faibles et que la solution est proche d'un minimum local. En choisissant une tolérance aussi faible, on encourage le solver à continuer l'optimisation jusqu'à ce qu'il atteigne une solution très précise, ce qui peut être important pour garantir que le modèle puisse apprendre à satisfaire la PDE de manière précise et faire des prédictions précises.
        history_size=10, # Nombre d'itérations dont le solver LBFGS garde l'historique pour construire l'approximation de la matrice inverse du Hessien. En utilisant history_size=10, on permet au solver de conserver les 10 dernières itérations d'optimisation, ce qui peut aider à améliorer la convergence du solver en fournissant plus d'informations sur la géométrie de la fonction de perte dans l'espace des paramètres. Cependant, cela peut également augmenter la mémoire utilisée par le solver, donc il est important de trouver un équilibre entre la taille de l'historique et les ressources disponibles.
        linesearch="zoom", # Stratégie de recherche linéaire utilisée par le solver LBFGS pour trouver le pas optimal à chaque itération d'optimisation. En utilisant linesearch="zoom", on permet au solver de rechercher de manière efficace le pas optimal en zoomant sur une intervalle de pas qui satisfait les conditions de Wolfe, ce qui peut aider à améliorer la convergence du solver et à trouver des solutions plus rapidement. Cependant, cela peut également augmenter le temps nécessaire pour chaque itération d'optimisation, donc il est important de trouver un équilibre entre la qualité du pas trouvé et le temps d'exécution global de l'entraînement.
        jit=True, # En utilisant jit=True, on permet au solver LBFGS de compiler la fonction de perte avec JAX pour qu'elle puisse être exécutée de manière plus efficace sur le matériel disponible (CPU ou GPU), ce qui peut accélérer considérablement les calculs pendant l'entraînement du modèle et permettre d'obtenir des résultats plus rapidement. En utilisant jit=True, on peut ainsi améliorer l'efficacité globale de l'entraînement du modèle en réduisant le temps nécessaire pour calculer la perte à chaque itération d'optimisation.
    )

    state = solver.init_state(train_params, data)

    # =====================================================
# ETAT INITIAL AVANT LA PREMIERE UPDATE
# =====================================================
    loss0, aux0 = compute_losses_fixed_jit(train_params, data)
    metrics0 = evaluate_common_metrics_jit(train_params, data)

    epoch_history.append(0)
    alpha_val_history.append(float(get_alpha(train_params)))
    loss_history.append(float(loss0))
    loss_ic_history.append(clean_small(float(aux0["loss_ic"])))
    loss_bc_history.append(clean_small(float(aux0["loss_bc"])))
    loss_data_history.append(float(aux0["loss_data"]))
    loss_phys_history.append(float(aux0["loss_phys"]))

    epoch_common_history.append(0)
    mse_solution_history.append(float(metrics0["mse_solution"]))
    loss_phys_common_history.append(float(metrics0["loss_phys_common"]))
    loss_total_common_history.append(float(metrics0["loss_total_common"]))

    T_snaps.append(np.array(metrics0["T_pred_common"]))
    epoch_snaps.append(0)

    # Buffers pour critère d'arrêt sur les 100 dernières itérations
    last100_mse_solution = deque(maxlen=WINDOW_STOP)
    last100_mse_phys = deque(maxlen=WINDOW_STOP)
    last100_mse_data = deque(maxlen=WINDOW_STOP)
    last100_mse_ic = deque(maxlen=WINDOW_STOP)
    last100_mse_bc = deque(maxlen=WINDOW_STOP)

    
    stop_reason = "max_iters"

    # =====================================================
    # LANCEMENT L-BFGS
    # =====================================================
    timing_before_train = time.time()

    converged = False
    failed = False
    final_loss = None
    num_iterations = 0

    run_progress = tqdm(
        total=N_epoch,
        desc=f"Run {run_idx}/{total_runs}",
        unit="it",
        leave=False,
        position=1
    )

    try:
        for k in range(1, N_epoch + 1):
            train_params, state = solver.update(train_params, state, data)
            num_iterations = k
            run_progress.update(1)

            metrics_stop = evaluate_common_metrics_jit(train_params, data)

            current_mse_solution = float(metrics_stop["mse_solution"])
            current_mse_phys = float(metrics_stop["loss_phys_common"])
            current_mse_data = float(metrics_stop["loss_data"])
            current_mse_ic = float(metrics_stop["loss_ic"])
            current_mse_bc = float(metrics_stop["loss_bc"])

            elapsed_run = time.time() - timing_before_train
            avg_time_per_iter = elapsed_run / k
            eta_run = avg_time_per_iter * (N_epoch - k)

            grad_val = float(grad_norm_jit(train_params, data))

            run_progress.set_postfix({
                "alpha": f"{float(get_alpha(train_params)):.4f}",
                "loss": f"{float(metrics_stop['loss']):.2e}",
                "data": f"{current_mse_data:.2e}",
                "phys": f"{current_mse_phys:.2e}",
                "grad_(LBFGS)": f"{grad_val:.2e}",
                "ETA": format_time(eta_run),
            })

            last100_mse_solution.append(current_mse_solution)
            last100_mse_phys.append(current_mse_phys)
            last100_mse_data.append(current_mse_data)
            last100_mse_ic.append(current_mse_ic)
            last100_mse_bc.append(current_mse_bc)

            if len(last100_mse_solution) == WINDOW_STOP:
                cond_solution = max(last100_mse_solution) < THRESH_MSE_SOLUTION
                cond_phys = max(last100_mse_phys) < THRESH_MSE_PHYS
                cond_data = max(last100_mse_data) < THRESH_MSE_DATA
                cond_ic = max(last100_mse_ic) < THRESH_MSE_IC
                cond_bc = max(last100_mse_bc) < THRESH_MSE_BC

                # if cond_solution and cond_phys and cond_data and cond_ic and cond_bc:
                if cond_phys and cond_data and cond_ic and cond_bc:
                    converged = True
                    final_loss = float(state.value)
                    stop_reason = "criteria_last_100_iterations"
                    print(
                        f" Arrêt anticipé à it={k} : "
                        f"sur les {WINDOW_STOP} dernières itérations, "
                        #f"MSE solution < {THRESH_MSE_SOLUTION:.1e}, "
                        f"MSE phys < {THRESH_MSE_PHYS:.1e}, "
                        f"MSE data < {THRESH_MSE_DATA:.1e}, "
                        f"MSE IC < {THRESH_MSE_IC:.1e}, "
                        f"MSE BC < {THRESH_MSE_BC:.1e}"
                    )
                    break

            if k % append_snap == 0:
                loss_val, aux = compute_losses_fixed_jit(train_params, data)

                epoch_history.append(k)
                alpha_val_history.append(float(get_alpha(train_params)))
                loss_history.append(float(loss_val))
                loss_ic_history.append(clean_small(float(aux["loss_ic"])))
                loss_bc_history.append(clean_small(float(aux["loss_bc"])))
                loss_data_history.append(float(aux["loss_data"]))
                loss_phys_history.append(float(aux["loss_phys"]))

            if k % append_snap_common == 0:
                metrics = evaluate_common_metrics_jit(train_params, data)

                epoch_common_history.append(k)
                mse_solution_history.append(float(metrics["mse_solution"]))
                loss_phys_common_history.append(float(metrics["loss_phys_common"]))
                loss_total_common_history.append(float(metrics["loss_total_common"]))

            if k % snap_every == 0:
                metrics = evaluate_common_metrics_jit(train_params, data)

                T_snaps.append(np.array(metrics["T_pred_common"]))
                epoch_snaps.append(k)

                print(
                    f"it {k} | "
                    f"total loss (PDE + data fixe): {float(metrics['loss']):.3e} | "
                    f"IC check: {clean_small(float(metrics['loss_ic'])):.3e} | "
                    f"BC check: {clean_small(float(metrics['loss_bc'])):.3e} | "
                    f"loss phys (native, fixe): {clean_small(float(metrics['loss_phys'])):.3e} | "
                    f"loss data: {clean_small(float(metrics['loss_data'])):.3e} | "
                    f"MSE solution grille: {clean_small(float(metrics['mse_solution'])):.3e} | "
                    f"loss phys (commune, grille): {clean_small(float(metrics['loss_phys_common'])):.3e} | "
                )

            if np.isnan(float(state.value)) or np.isinf(float(state.value)):
                failed = True
                final_loss = float(state.value)
                stop_reason = "nan_or_inf"
                print("Arrêt : NaN / Inf détecté dans l'objectif.")
                break

            if float(state.error) < 1e-9:
                converged = True
                final_loss = float(state.value)
                stop_reason = "lbfgs_converged"
                print("Arrêt : critère interne L-BFGS atteint.")
                break

    except Exception as e:
        print(f"Erreur pendant l'entraînement : {e}")
        failed = True
        stop_reason = "exception"

    finally:
        if final_loss is None:
            final_loss = float(state.value)
            converged = bool(float(state.error) < 1e-9)

        run_progress.close()

    total_time = time.time() - timing_before_train
    print(f" Temps total: {total_time:.1f}s")
    print(f"Converged: {converged}")
    print(f"Failed   : {failed}")
    print(f"Iterations L-BFGS: {num_iterations}")
    print(f"Total loss finale (PDE sur LHS + data sur pts expérimentaux) : {final_loss:.3e}")

    # =====================================================
    # TEST FINAL ALEATOIRE
    # =====================================================
    np.random.seed(1) # DIFFERENT SEED QUE POUR LHS, POUR AVOIR UN TEST FINAL DIFFERENT DES POINTS DE LA PDE
    t_final_phys = np.random.uniform(0.0, max_t, size=(N_rand_final, 1)).astype(np.float64)
    x_final_phys = np.random.uniform(0.0, L_barre, size=(N_rand_final, 1)).astype(np.float64)

    t_final_phys = jnp.array(t_final_phys, dtype=DTYPE)
    x_final_phys = jnp.array(x_final_phys, dtype=DTYPE)

    t_final_net, x_final_net = phys_to_net(t_final_phys, x_final_phys)
    tx_final_net = jnp.concatenate([t_final_net, x_final_net], axis=1)

    T_pred_final = model_apply(train_params["nn"], tx_final_net)
    T_true_final = exact_solution(t_final_phys, x_final_phys, alpha_theo, L_tf)

    mse_final = jnp.mean((T_pred_final - T_true_final) ** 2)

    r_test = physics_residual(train_params, tx_final_net)
    mse_residual_native = jnp.mean(r_test ** 2)
    r_phys = residual_native_to_physical(r_test, mode)
    mse_residual_phys = jnp.mean(r_phys ** 2)

    # Vérifications IC / BC finales
    T_ic_final = model_apply(train_params["nn"], data["tx_ic_net"])
    mse_ic_final = jnp.mean((T_ic_final - data["T_ic"]) ** 2)

    T_bc0_final = model_apply(train_params["nn"], data["tx_bc0_net"])
    T_bc1_final = model_apply(train_params["nn"], data["tx_bc1_net"])
    mse_bc_final = jnp.mean(T_bc0_final ** 2) + jnp.mean(T_bc1_final ** 2)

    # Véridication finale data
    T_pred_obs_final = model_apply(train_params["nn"], data["tx_data_net"])
    mse_data_final = jnp.mean((T_pred_obs_final - T_obs_jax) ** 2)

    # Alpha final
    alpha_final = get_alpha(train_params)
     
    print(f"alpha final appris                        : {float(alpha_final):.6f}")
    print(f"MSE solution finale (pts aléatoires)       : {float(mse_final):.3e}")
    print(f"MSE résidu native finale (pts aléatoires)  : {float(mse_residual_native):.3e}")
    print(f"MSE résidu physique final (pts aléatoires) : {float(mse_residual_phys):.3e}")
    print(f"MSE data finale (pts dataset)                   : {float(mse_data_final):.3e}")
    print(f"MSE IC finale (check)                      : {clean_small(float(mse_ic_final)):.3e}")
    print(f"MSE BC finale (check)                      : {clean_small(float(mse_bc_final)):.3e}")

    metrics_final_plot = evaluate_common_metrics_jit(train_params, data)

    if len(T_snaps) == 0:
       T_snaps.append(np.array(metrics_final_plot["T_pred_common"]))
       epoch_snaps.append(num_iterations)
    else:
       T_snaps[-1] = np.array(metrics_final_plot["T_pred_common"])
       epoch_snaps[-1] = num_iterations

    save_run_plots(
        res={
            "title": title,
            "epoch_history": epoch_history,
            "epoch_common_history": epoch_common_history,
            "alpha_fit": alpha_fit,
            "alpha_val_history": alpha_val_history,
            "loss_history": loss_history,
            "loss_ic_history": loss_ic_history,
            "loss_bc_history": loss_bc_history,
            "loss_phys_history": loss_phys_history,
            "loss_data_history": loss_data_history,
            "loss_phys_common_history": loss_phys_common_history,
            "loss_total_common_history": loss_total_common_history,
            "mse_solution_history": mse_solution_history,
            "T_snaps": T_snaps,
            "T_final_pred": np.array(metrics_final_plot["T_pred_common"]),
            "T_true": np.array(T_true),
            "t_test_phys": np.array(t_test_phys),
            "x_test_phys": np.array(x_test_phys),
        },
        run_dir=run_dir,
        L_barre=L_barre,
        max_t=max_t,
        N_test=N_test,
        alpha_theo=alpha_theo,
        mode=mode,
    )
    

    return {
        "title": title,
        "mode": mode,
        "params": train_params["nn"],    
        "alpha_final": float(get_alpha(train_params)),
        "epoch_history": epoch_history,
        "epoch_common_history": epoch_common_history,
        "alpha_val_history": alpha_val_history,
        "loss_history": loss_history,
        "loss_ic_history": loss_ic_history,
        "loss_bc_history": loss_bc_history,
        "loss_phys_history": loss_phys_history,
        "loss_data_history": loss_data_history,
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
        "mse_data_final": float(mse_data_final),
        "mse_ic_final": float(mse_ic_final),
        "mse_bc_final": float(mse_bc_final),
        "num_iterations": int(num_iterations),
        "total_loss_finale": float(final_loss),
        "lambda_data": lambda_data_var,
        "seed": seed_var,
        "pts coloc": n_coloc,
        "converged": bool(converged),
        "grad_LBFGS_final": float(grad_norm_jit(train_params, data)),
        "failed (Nan ou Inf)": bool(failed),
        "stop_reason": stop_reason,
        "run_dir": str(run_dir),
       # "criteria_solution_ok": bool(float(mse_final) < THRESH_MSE_SOLUTION),
        "criteria_phys_ok": bool(float(mse_residual_phys) < THRESH_MSE_PHYS),
        "criteria_data_ok": bool(float(mse_data_final) < THRESH_MSE_DATA),
        "criteria_ic_ok": bool(float(mse_ic_final) < THRESH_MSE_IC),
        "criteria_bc_ok": bool(float(mse_bc_final) < THRESH_MSE_BC),
        "all_final_mse_ok": bool(
    # (float(mse_final) < THRESH_MSE_SOLUTION) and
    (float(mse_residual_phys) < THRESH_MSE_PHYS) and
    (float(mse_data_final) < THRESH_MSE_DATA) and
    (float(mse_ic_final) < THRESH_MSE_IC) and
    (float(mse_bc_final) < THRESH_MSE_BC)
),
    }


