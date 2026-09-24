from drive_utils import create_subst_drive, remove_subst_drive
from pathlib import Path
BASE_PATH = str(Path(__file__).resolve().parent) # Dossier du script, calculé dynamiquement pour rester portable d'une machine/d'un dossier à l'autre.
DRIVE = create_subst_drive(BASE_PATH) # Creer un raccourci pour eviter les noms trop longs avec les chemins complets, ce qui peut causer des erreurs de type "OSError: [Errno 36] File name too long" lors de la sauvegarde des fichiers ou de l'accès aux répertoires. En utilisant create_subst_drive, on peut créer un lecteur virtuel pointant vers le répertoire de travail, ce qui permet d'utiliser des chemins plus courts et d'éviter les problèmes liés à la longueur des chemins dans les systèmes de fichiers.

# ===== Standard =====
import time # Pour calculer le temps de simulation
timing_INITIAL = time.time()
import argparse # Pour récupérer les argument du ficher .bat
import shutil # pour gérer les fichiers et les répertoires, notamment pour supprimer les caches de compilation JAX entre les runs

# ===== Data / numerics =====
import numpy as np
import pandas as pd

# ===== Plot =====
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg") # Empeche les fenetres des plot de s'ouvrir et enregistre direct les plots.
# ===== JAX =====
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.experimental.compilation_cache import compilation_cache as cc # Pour garder en cache les compilations JAX entre les runs, ce qui accélère beaucoup les exécutions suivantes après la première.

# ===== SciPy (fit alpha classique) =====
from scipy.optimize import least_squares # Pour le fit classique de alpha du dataset (avec la méthode des moindres carrés)

# ===== Progress bar =====
from tqdm.auto import tqdm # Barre de progression pour les boucles d'entraînement

# ===== Ton code =====
from generate_noisy_dataset import noisy_data_set
from training import run_experiment
from summary import save_summary_outputs

from utils import clean_small, format_time, make_run_dir

from model import (
    get_activation,
    make_model_params,
    init_alpha_raw_from_alpha,
    model_apply_raw,
)

cc.set_cache_dir(str(Path(DRIVE) / "jax_cache")) # PERMET DE GARDER EN CACHE LES COMPILATIONS JAX ENTRE LES RUNS, CE QUI ACCELERE BEAUCOUP LES EXÉCUTIONS SUIVANTES APRÈS LA PREMIÈRE.

mpl.rcParams["figure.max_open_warning"] = 100


# CHOIX DU TYPE NUMERIQUE (FLOAT 64 POUR MEILLEURE PRECISION ET COMPATIBILTE AVEC JAX)
DTYPE = jnp.float64

# RÉCUPÉRER LES ARGUMENTS DU FICHIER .BAT
# parser = argparse.ArgumentParser(description="Génération dataset PINN bruité")

# parser.add_argument("--alpha_theo", type=float, default=0.18,
#                     help="Valeur théorique de alpha")
# parser.add_argument("--noise_level", type=float, default=0.05,
#                     help="Niveau de bruit (fraction de std du signal)")
# parser.add_argument("--N_data_points", type=int, default=500,
#                     help="Nombre de points")
# parser.add_argument("--dataset_seed", type=int, default=0,
#                     help="Seed pour génération du dataset bruité")
# parser.add_argument("--seed", type=int, default=0,
#                   help="Seed pour reproductibilité")

# args = parser.parse_args()

# alpha_theo = args.alpha_theo # Coeff de diffusion théorique utilisé pour générer le dataset
# noise_level = args.noise_level # Calcul du taux de bruit pour générer le dataset bruité. Pourcentage de la variabilité par rapport à la moyenne de T_data (ex 0.05 = 5% de sigma(T))
# data_set_seed = args.dataset_seed # Seed pour générer des dataset différents avec des points (t, x) tirés aléatoirement, avec la température observée T_obs correspondante
# N_data_points = args.N_data_points # Nombre de points du dataset généré


# =========================================================
# PARAMETRES
# =========================================================

trial = "5_poids_data_sqrt" # Numéro de la trial, pour différencier les dossiers de résultats globaux entre différentes séries d'expériences. En utilisant trial, on peut organiser les résultats de différentes séries d'expériences dans des dossiers séparés, ce qui facilite la comparaison des résultats obtenus avec différents paramètres ou configurations d'entraînement. En ajustant le numéro de trial, on peut également suivre l'évolution des résultats au fil du temps et identifier les tendances ou les améliorations apportées par les différentes stratégies d'entraînement testées dans les expériences.
# PARAMETRES DE ROBUSTESSE DE ALPHA

N_data_points = [25] # Nombre de points du dataset généré
alpha_theo = [0.01, 0.18, 0.5] # Coeff de diffusion théorique utilisé pour générer le dataset (initialement 0.18)
noise_level = [0.05, 0.1, 0.2] # Calcul du taux de bruit pour générer le dataset bruité. Pourcentage de la variabilité par rapport à la moyenne de T_data (ex 0.05 = 5% de sigma(T))
data_set_seed = [0, 1] # Seed pour générer des dataset différents avec des points (t, x) tirés aléatoirement, avec la température observée T_obs correspondante

# CRITERES D'ARRET DE SIMULATION ANTICIPE

WINDOW_STOP = 100 # Nombre d'itérations consécutives pendant lesquelles on vérifie les critères d'arrêt anticipé (100 ici). En utilisant WINDOW_STOP, on peut s'assurer que les critères d'arrêt sont vérifiés de manière stable sur une fenêtre d'itérations, ce qui permet d'éviter les arrêts prématurés dus à des fluctuations temporaires des métriques pendant l'entraînement du modèle. En vérifiant les critères d'arrêt sur une fenêtre de plusieurs itérations, on peut garantir que le modèle a suffisamment de temps pour converger vers une solution optimale avant de décider d'arrêter l'entraînement, ce qui peut améliorer la performance finale du modèle et sa capacité à satisfaire la PDE de manière précise.
THRESH_MSE_SOLUTION = 5e-6 # Seuil MSE solution. Pas utilisé ici pour dans le pb reel parce qu'on connait pas la solution exacte
THRESH_MSE_PHYS = 5e-5 # Seuil MSE résidu PDE
THRESH_MSE_DATA = None #. 5e-4 # Seuil MSE données
THRESH_MSE_IC = 1e-16 # Seuil MSE conditions initiales, très strict pour hard IC
THRESH_MSE_BC = 1e-16 # Seuil MSE conditions aux bords, très strict pour hard BC

# PARAMETRES GENERAUX
N_epoch = 3000 # Default 5000. Nombre maximal d'itérations L-BFGS

max_t = 1.5 # Temps maximal de la simulation (t = 1.5 pour consistence avec le dataset généré avec max_t = 1.5)
L_barre = 1.0 # Longeur de la barre (L_barre = 1.0 pour consistence avec le dataset)

T_threshold_factor = 5  # Critere de seuil du signal T, en facteur multiplicatif du bruit pour ponderation de la loss data avec loss_mod_signal_only, pour que les points où T est petit aient moins d'importance dans la loss data, ce qui peut aider à améliorer la précision du modèle pour les points où le signal est plus fort et ainsi mieux satisfaire la PDE. En ajustant T_threshold, on peut contrôler à partir de quelle valeur de T_obs le poids de la loss data commence à diminuer, ce qui permet de trouver un équilibre entre accorder suffisamment d'importance aux points où le signal est fort tout en évitant que les points où le signal est faible n'influencent trop l'entraînement du modèle.

window_noise = 5 # Taille de la fenêtre pour l'estimation locale du bruit, utilisée dans la fonction estimate_noise_local_affine_1d pour estimer le niveau de bruit local à partir des données observées. En utilisant une fenêtre de taille 5, on peut effectuer des ajustements locaux sur les données pour estimer le bruit de manière plus précise, ce qui peut aider à ajuster le poids de la loss data en fonction du niveau de bruit estimé pour chaque point, améliorant ainsi la précision du modèle pour les points où les prédictions sont plus grandes et aidant à mieux satisfaire la PDE.
method_noise = "std" # Méthode pour estimer le bruit local, "std" pour l'écart type standard, ou "mad" pour la médiane absolue des écarts à la médiane. En utilisant "std", on peut obtenir une estimation du bruit basée sur l'écart type des résidus locaux, tandis qu'en utilisant "mad", on peut obtenir une estimation plus robuste du bruit qui est moins sensible aux valeurs aberrantes. Le choix de la méthode dépend du niveau de bruit dans les données et de la manière dont on souhaite pondérer les différentes parties de la loss data pour guider l'apprentissage du modèle.
crit_noise = 1.0 # Facteur multiplicatif pour ajuster l'estimation du bruit dans le poids de la loss data, qui est calculé en fonction des prédictions du modèle T_pred_obs. En ajustant crit_noise, on peut accorder plus ou moins d'importance aux points où les prédictions du modèle sont plus grandes, ce qui peut aider à améliorer la précision du modèle pour ces points et ainsi mieux satisfaire la PDE. Le choix de crit_noise dépend du niveau de bruit dans les données et de la manière dont on souhaite pondérer les différentes parties de la loss data pour guider l'apprentissage du modèle.

alpha_inits = [0.05, 1.0] # Valeur initiale de alpha pour l'optimisation, dans le prob inverse
# lambdas_data = [0.1, 1, 10] # Differents lambda data dans la loss
lamb_mul = [0.8] # Facteurs multiplicatifs pour ajuster lambda_data en fonction du niveau de bruit estimé, pour tester différents poids de la loss data dans les expériences et trouver le meilleur compromis entre accorder suffisamment d'importance aux points où les prédictions du modèle sont plus grandes tout en évitant que les points où les prédictions sont faibles n'influencent trop l'entraînement du modèle, ce qui peut aider à améliorer la précision du modèle pour les points où le signal est plus fort et ainsi mieux satisfaire la PDE. En ajustant lamb_mul, on peut trouver le meilleur facteur multiplicatif pour lambda_data en fonction du niveau de bruit dans les données et de la manière dont on souhaite pondérer les différentes parties de la loss data pour guider l'apprentissage du modèle.
lambdas_data = [1] # Differents lambda data dans la loss
widths_nn = [16] # Nombre de neurones
depths_nn = [3] # Profondeur du réseau
lhs_seeds = [0] # Différentes seed pour le LHS (attention n'affecte pas l'initialisation des poids)
colocs = [500] # Nombres de pts de coloc différents pour la PDE


f_activation = "tanh" # Default tanh. Fnction d'activation des neurones

# Type de pondération dans loss data
# loss_mode = "data_loss_non_mod" # (pas de ponderation)
# loss_mode = "data_loss_mod_signal_only" # (ponderation par le signal uniquement, avec (1 + T_pred_obs^2) pour ajuster le poids de la loss data en fonction des prédictions du modèle, ce qui peut aider à améliorer la précision du modèle pour les points où les prédictions sont plus grandes et ainsi mieux satisfaire la PDE)
loss_mode = "data_loss_mod_alpha_info" # Ponderation avec des poids plus for la ou alpha est plus informatif
                                            
# PARAMETRES ENTRAINEMENT ET EVALUATION FINALE

# lr_w = 0.01 # Learning rate pour poids W,b. Default 0.001 (meilleur), increase learning rate to speed up convergence, but can cause instability if too high. Decrease learning rate to 0.00001 for more stable training and better convergence towards la solution exacte, especially in the later stages of training when the model is close to the optimal solution. A lower learning rate allows for finer adjustments of the model weights, which can help to reduce oscillations around the optimal solution and improve the final accuracy of the model. However, it may also increase the time required for training, so it's important to find a balance between convergence speed and stability.
# lr_alpha = 0.1 # Learning rate pour alpha, Default 0.001 (meilleur), increase learning rate to speed up convergence, but can cause instability if too high. Decrease learning rate to 0.00001 for more stable training and better convergence towards la solution exacte, especially in the later stages of training when the model is close to the optimal solution. A lower learning rate allows for finer adjustments of alpha, which can help to reduce oscillations around the optimal solution and improve the final accuracy of the model. However, it may also increase the time required for training, so it's important to find a balance between convergence speed and stability.

# Points de colocation pour les IC/BC (juste pour monitoring en contraintes dures, pas besoin de beaucoup de points)
N_ic = 100 # Pts de colocs IC
N_bc = 100 # Pts de coloc BC

# Pondération des loss
# lambda_ic = 0.0
# lambda_bc = 0.0
lambda_phys = 1.0
#lambda_data = 100. Passé en argument des runs pour tester différents lambda data

# Grille de test pour évaluation pdt l'entraînement
N_test = 50 # Nombre de points de test (sur grille fixe) pour évaluer l'entrainement. 

# Fréquence d'enregistrement / affichage
snap_every = max(1, N_epoch // 10)
append_snap = 1 # Fréquence d'enregistrement des snapshots de la solution pendant l'entraînement, pour visualiser l'évolution de la solution au cours de l'entraînement. En utilisant append_snap, on peut contrôler la fréquence à laquelle les snapshots sont enregistrés, ce qui permet de suivre l'évolution de la solution à différents stades de l'entraînement et d'identifier les moments où le modèle commence à converger vers une solution satisfaisante ou à rencontrer des difficultés pour satisfaire la PDE. En ajustant append_snap, on peut trouver un équilibre entre la quantité d'informations enregistrées et la clarté des visualisations de l'évolution de la solution pendant l'entraînement.
# append_snap_common = max(1, snap_every // 10)
append_snap_common = 1 # Fréquence d'enregistrement des snapshots de la solution pendant l'entraînement pour les runs avec reconversion des loss en physique pour les normalisations.

# Nombre de points pour évaluation finale après entraînement (aléatoires)
N_rand_final = 10000

# Fo = alpha_val * max_t / (L_barre ** 2) # Nombre de fourrier
# Fo_tf = jnp.array(Fo, dtype=DTYPE)

print(f"temps max de simulation max_t = {max_t}")
print(f"longueur de la barre L = {L_barre}")
print(f"diffusivité thermique initial alpha = {alpha_inits}")
# print(f"Fourier number Fo = alpha * t_max / L^2 = {Fo:.6f}")

total_runs = ( # Nombre total de runs à effectuer, calculé en fonction du nombre de combinaisons de paramètres testées dans les expériences. En utilisant total_runs, on peut suivre la progression globale des expériences et estimer le temps restant pour compléter toutes les expériences en fonction du temps moyen par run et du nombre de runs restants.
    len(alpha_theo)
    * len(noise_level)
    * len(N_data_points)
    * len(data_set_seed)
    * len(alpha_inits)
    * len(lambdas_data)
    * len(lamb_mul)
    * len(widths_nn)
    * len(depths_nn)
    * len(lhs_seeds)
    * len(colocs)
)

print(f"Nombre total de runs = {total_runs}")

### Conversion des paramètres en types JAX (pour éviter des erreurs de type pendant l'entraînement)
pi = jnp.array(np.pi, dtype=DTYPE)
L_tf = jnp.array(L_barre, dtype=DTYPE)
tmax_tf = jnp.array(max_t, dtype=DTYPE)


# =========================================================
# RUN DES EXPERIENCES
# =========================================================

activation_fn = get_activation(f_activation) # Convertit la fonction d'activation en jax


run_counter = 0 # Compteur de runs pour suivre le nombre d'expériences effectuées, ce qui est utile pour afficher la progression globale dans la barre de progression et pour calculer le temps moyen par run et l'ETA (temps restant estimé) en fonction du nombre total de runs. En utilisant run_counter, on peut également associer les résultats de chaque expérience à un numéro de run spécifique, ce qui facilite l'organisation et l'analyse des résultats obtenus pour chaque combinaison de paramètres testée dans les expériences.
timing_global_start = time.time()

pbar = tqdm(total=total_runs, desc="Progression globale", unit="run", position=0) # Barre de progression globale pour suivre l'avancement de toutes les expériences

all_results = []
dataset_results = []
OUTPUT_ROOT = None
dataset_name = None

GLOBAL_OUTPUT_ROOT = Path(
    f"{DRIVE}\\results_robustesse_pb_reel_1D_inverse_noisy_data_HARD_IC_BC_GLOBAL_{f_activation}_{loss_mode}_mse_DATA_mod_trial_{trial}"
)
GLOBAL_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
try:
    for alpha_th in alpha_theo:
        for noise_lvl in noise_level:
            for N_data_pts in N_data_points:
                for data_seed in data_set_seed:

                    df, N_data_pts_mod = noisy_data_set(data_seed, max_t, L_barre, N_data_pts, noise_lvl, alpha_th)

                    N_data_pts_initial = N_data_pts
                    N_data_pts_final = N_data_pts_mod

                    dataset_csv_path = GLOBAL_OUTPUT_ROOT / f"{N_data_pts_final}_pts_final_{N_data_pts_initial}_pts_initial_alpha_theo_{alpha_th}_noise_level_{noise_lvl}_set_seed_{data_seed}.csv"
                    df.to_csv(dataset_csv_path, index=False)

                    dataset_path = dataset_csv_path
                    dataset_name = dataset_path.stem

                    df = pd.read_csv(dataset_csv_path)

                    OUTPUT_ROOT = GLOBAL_OUTPUT_ROOT / dataset_name
                        
                    print(f"Dataset initial : {N_data_pts_initial} points")
                    print(f"Dataset final : {N_data_pts_final} points")

                    x_data = df["x"].values
                    t_data = df["t"].values

                    T_clean = df["T_clean"].values
                    T_obs = df["T_obs"].values

                    sigma_est = noise_lvl * np.std(T_clean)
                    
                    lambda_data_ref = (0.1 / (noise_lvl + 1e-12))**3   # Valeur de lambda_ref dérivée empiriquement à partir des résultats
                    lambdas_data = [lambda_data_ref * m for m in lamb_mul]   # On fait varier lambda autour de lambda_ref

                    T_threshold = T_threshold_factor * sigma_est # Pour loss mod signal_only

                    THRESH_MSE_DATA = sigma_est**2  # Critère MSE Data intégrant le niveau de bruit
                    
                    x_data_jax = jnp.array(x_data[:, None], dtype=DTYPE)
                    t_data_jax = jnp.array(t_data[:, None], dtype=DTYPE)
                    T_obs_jax  = jnp.array(T_obs[:, None], dtype=DTYPE)

                    def residual(alpha):
                        a = alpha[0]
                        T_analy_data = np.sin(np.pi * x_data) * np.exp(-a * np.pi**2 * t_data)
                        return T_analy_data - T_obs

                    res = least_squares(residual, x0=[0.2], bounds=(0, np.inf))
                    alpha_fit = res.x[0] ## Fit les données pour obtenir la valeur de alpha avec la méthode des moindres carrés

                    C = (np.pi / L_barre) ** 2

                    # THRESH_MSE_PHYS = 20.0 * (alpha_fit * C * sigma_est)**2 # Critère MSE Phys intégrant le niveau de bruit
                    
                    # Calcul la dérivé de la solution exacte par rapport à alpha, pour connaitre les zones (x,t) les plus informatives pour alpha (xmax = Lbarre/2) et tmax = (L/pi*)**2 x (1/alpha) 
                    poids_alpha_info = np.abs(   # PAS POUR LE PROBLEME REEL !!!! parce que on connait pas en principe alpha_fit dérivé de la solution exacte.
                        C * t_data * np.sin(np.pi * x_data / L_barre) # Ici les poids de la loss data correspondent à la courbe de dT/dalpha, obtenue pour alpha_fit. 
                        * np.exp(-alpha_fit * C * t_data)
                    )

                    poids_alpha_info_jax = jnp.array(poids_alpha_info[:, None], dtype=DTYPE)
                    
                    # # Test avec en prenant les poids au carré pour accentuer encore plus les différences de poids entre les zones plus ou moins informatives pour alpha, pour voir si ça aide à améliorer la précision du modèle pour les points où le signal est plus fort et ainsi mieux satisfaire la PDE.
                    poids_alpha_info_jax = poids_alpha_info_jax**(1/2) # On peut aussi essayer d'augmenter encore plus les poids dans les zones les plus informatives en prenant le carré de la dérivée, ce qui accentue encore plus les différences de poids entre les zones plus ou moins informatives pour alpha.

                    print("alpha_theorique =", alpha_th)
                    print("alpha_fit =", alpha_fit)

                    noise_estimation = np.var(T_obs)  # provisoire si tu n'as pas encore branché l'estimation affine

                    data_arrays = {
                        "x_data_jax": x_data_jax,
                        "t_data_jax": t_data_jax,
                        "T_obs_jax": T_obs_jax, # Points de donnés bruités
                        "poids_alpha_info_jax": poids_alpha_info_jax,
                    }

                    config = {
                        "DTYPE": DTYPE,
                        "L_barre": L_barre,
                        "max_t": max_t,
                        "L_tf": L_tf,
                        "alpha_theo": alpha_th,
                        "alpha_fit": alpha_fit,
                        "N_ic": N_ic,
                        "N_bc": N_bc,
                        "N_test": N_test,
                        "N_epoch": N_epoch,
                        "N_rand_final": N_rand_final,
                        "snap_every": snap_every,
                        "append_snap": append_snap,
                        "append_snap_common": append_snap_common,
                        "WINDOW_STOP": WINDOW_STOP,
                        "THRESH_MSE_SOLUTION": THRESH_MSE_SOLUTION,
                        "THRESH_MSE_PHYS": THRESH_MSE_PHYS,
                        "THRESH_MSE_DATA": THRESH_MSE_DATA,
                        "THRESH_MSE_IC": THRESH_MSE_IC,
                        "THRESH_MSE_BC": THRESH_MSE_BC,
                        "lambda_phys": lambda_phys,
                        "T_threshold": T_threshold,
                        "window_noise": window_noise,
                        "method_noise": method_noise,
                        "crit_noise": crit_noise,
                        "noise_estimation": noise_estimation,
                        "loss_mode": loss_mode, 
                    }

                    if OUTPUT_ROOT.exists():
                        shutil.rmtree(OUTPUT_ROOT)
                    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

                    dataset_results = []
                    
                    for alpha0 in alpha_inits:
                        for lam in lambdas_data:
                            for w in widths_nn:
                                for d in depths_nn:
                                    for s in lhs_seeds:
                                        for c in colocs:

                                            run_counter += 1
                                            run_start_time = time.time()

                                            base_train_params = {
                                                "nn": make_model_params(width=w, depth=d),
                                                #"alpha_raw": inverse_softplus(jnp.array(alpha0, dtype=DTYPE) - alpha_min).astype(DTYPE),
                                                "alpha_raw": init_alpha_raw_from_alpha(alpha0).astype(DTYPE), # alpha initiale pour l'optimisation, alpha = alpha_min + beta^2, pour assurer alpha > 0 et ralentir la convergence de alpha pour eviter d'apprendre la solution alpha = 0
                                            }
                                                                        
                                            title = f" loss_mode={loss_mode} | N_epochs_max = {N_epoch} | data_pts_initial={N_data_pts_initial} | data_pts_final={N_data_pts_final} | alpha_theo={alpha_th} | noise_level={noise_lvl} |  data_seed={data_seed} | alpha_init={alpha0} | lam={lam} | w={w} | d={d} | lhs_seed={s} | coloc={c} | act={f_activation}"

                                            run_dir = make_run_dir(
                                                OUTPUT_ROOT,
                                                alpha0=alpha0,
                                                lam=f"{lam:.1f}",
                                                w=w,
                                                d=d,
                                                s=s,
                                                c=c,
                                                act=f_activation,
                                            )

                                            results = run_experiment(
                                            mode="non_normalized",
                                            title=title,
                                            base_train_params=base_train_params,
                                            lambda_data_var=lam,
                                            seed_var=s,
                                            n_coloc=c,
                                            run_idx=run_counter,
                                            total_runs=total_runs,
                                            run_dir=run_dir,
                                            data_arrays=data_arrays,
                                            config=config,
                                            model_apply_raw=model_apply_raw,
                                            activation_fn=activation_fn,
                                            )
                                            
                                            results["thresh_mse_data"] = THRESH_MSE_DATA
                                            results["thresh_mse_phys"] = THRESH_MSE_PHYS
                                            results["alpha0"] = alpha0
                                            results["alpha_fit"] = alpha_fit
                                            results["alpha_theo"] = alpha_th
                                            results["noise_level"] = noise_lvl
                                            results["N_data_points_initial"] = N_data_pts_initial
                                            results["N_data_points_final"] = N_data_pts_final
                                            results["N_removed"] = N_data_pts_initial - N_data_pts_final
                                            results["kept_ratio"] = N_data_pts_final / N_data_pts_initial
                                            results["removed_ratio"] = 1.0 - results["kept_ratio"]
                                            results["dataset_seed"] = data_seed
                                            results["width"] = w
                                            results["depth"] = d
                                            results["lhs_seed"] = s
                                            results["coloc"] = c
                                            results["activation"] = f_activation
                                            results["loss_mode"] = loss_mode
                                            results["poids_alpha_info_min"] = float(np.min(poids_alpha_info))
                                            results["poids_alpha_info_max"] = float(np.max(poids_alpha_info))
                                            results["poids_alpha_info_mean"] = float(np.mean(poids_alpha_info))
                                            results["poids_alpha_info_std"] = float(np.std(poids_alpha_info))

                                            dataset_results.append(results)
                                            all_results.append(results)

                                            # CHECKPOINT APRES CHAQUE RUN
                                            save_summary_outputs(
                                            all_results=dataset_results,
                                            output_root=OUTPUT_ROOT,
                                            dataset_name=dataset_name,
                                            f_activation=f_activation,
                                            loss_mode=loss_mode,
                                            thresh_mse_phys=THRESH_MSE_PHYS,
                                            thresh_mse_data=THRESH_MSE_DATA,
                                            thresh_mse_ic=THRESH_MSE_IC,
                                            thresh_mse_bc=THRESH_MSE_BC,
                                            )

                                            save_summary_outputs(
                                            all_results=all_results,
                                            output_root= GLOBAL_OUTPUT_ROOT,
                                            dataset_name="GLOBAL",
                                            f_activation=f_activation,
                                            loss_mode=loss_mode,
                                            thresh_mse_phys=THRESH_MSE_PHYS,
                                            thresh_mse_data=THRESH_MSE_DATA,
                                            thresh_mse_ic=THRESH_MSE_IC,
                                            thresh_mse_bc=THRESH_MSE_BC,
)

                                            run_time = time.time() - run_start_time
                                            elapsed_total = time.time() - timing_global_start

                                            avg_time_per_run = elapsed_total / run_counter
                                            remaining_runs = total_runs - run_counter
                                            eta_seconds = avg_time_per_run * remaining_runs

                                            pbar.update(1)
                                            pbar.set_postfix({
                                                "run": f"{run_counter}/{total_runs}",
                                                "last": format_time(run_time),
                                                "avg": format_time(avg_time_per_run),
                                                "ETA": format_time(eta_seconds),
                                            })

except KeyboardInterrupt:
    print("\nInterruption manuelle détectée. Sauvegarde des résultats déjà terminés...")

    if OUTPUT_ROOT is not None and dataset_name is not None:
        save_summary_outputs(
            all_results=dataset_results,
            output_root=OUTPUT_ROOT,
            dataset_name=dataset_name,
            f_activation=f_activation,
            loss_mode=loss_mode,
            thresh_mse_phys=THRESH_MSE_PHYS,
            thresh_mse_data=THRESH_MSE_DATA,
            thresh_mse_ic=THRESH_MSE_IC,
            thresh_mse_bc=THRESH_MSE_BC,
        )

    if GLOBAL_OUTPUT_ROOT is not None:
        save_summary_outputs(
            all_results=all_results,
            output_root=GLOBAL_OUTPUT_ROOT,
            dataset_name="GLOBAL",
            f_activation=f_activation,
            loss_mode=loss_mode,
            thresh_mse_phys=THRESH_MSE_PHYS,
            thresh_mse_data=THRESH_MSE_DATA,
            thresh_mse_ic=THRESH_MSE_IC,
            thresh_mse_bc=THRESH_MSE_BC,
        )

except Exception as e:
    print(f"\nErreur globale détectée : {e}")

    if OUTPUT_ROOT is not None and dataset_name is not None:
        save_summary_outputs(
            all_results=dataset_results,
            output_root=OUTPUT_ROOT,
            dataset_name=dataset_name,
            f_activation=f_activation,
            loss_mode=loss_mode,
            thresh_mse_phys=THRESH_MSE_PHYS,
            thresh_mse_data=THRESH_MSE_DATA,
            thresh_mse_ic=THRESH_MSE_IC,
            thresh_mse_bc=THRESH_MSE_BC,
        )

    if GLOBAL_OUTPUT_ROOT is not None:
        save_summary_outputs(
            all_results=all_results,
            output_root=GLOBAL_OUTPUT_ROOT,
            dataset_name="GLOBAL",
            f_activation=f_activation,
            loss_mode=loss_mode,
            thresh_mse_phys=THRESH_MSE_PHYS,
            thresh_mse_data=THRESH_MSE_DATA,
            thresh_mse_ic=THRESH_MSE_IC,
            thresh_mse_bc=THRESH_MSE_BC,
        )
    raise

else:
    if OUTPUT_ROOT is not None and dataset_name is not None:
        save_summary_outputs(
            all_results=dataset_results,
            output_root=OUTPUT_ROOT,
            dataset_name=dataset_name,
            f_activation=f_activation,
            loss_mode=loss_mode,
            thresh_mse_phys=THRESH_MSE_PHYS,
            thresh_mse_data=THRESH_MSE_DATA,
            thresh_mse_ic=THRESH_MSE_IC,
            thresh_mse_bc=THRESH_MSE_BC,
        )

    if GLOBAL_OUTPUT_ROOT is not None:
        save_summary_outputs(
            all_results=all_results,
            output_root=GLOBAL_OUTPUT_ROOT,
            dataset_name="GLOBAL",
            f_activation=f_activation,
            loss_mode=loss_mode,
            thresh_mse_phys=THRESH_MSE_PHYS,
            thresh_mse_data=THRESH_MSE_DATA,
            thresh_mse_ic=THRESH_MSE_IC,
            thresh_mse_bc=THRESH_MSE_BC,
        )

finally:
    pbar.close()
    remove_subst_drive(DRIVE)
    plt.close("all")

# =========================================================
# RESUME NUMERIQUE DES RUNS
# =========================================================
print("\n" + "=" * 300)
print("RESUME FINAL (points aléatoires)")
print("=" * 300)
for res in all_results:
    print(
        f"{res['title']:<45} | "
        f"MSE sol (aléatoire) = {res['mse_final']:.3e} | "
        f"MSE residu phys (aléatoire) = {res['mse_residual_phys']:.3e} | "
        f"MSE data check = {res['mse_data_final']:.3e} | "
        f"MSE IC check = {clean_small(res['mse_ic_final']):.3e} | "
        f"MSE BC check = {clean_small(res['mse_bc_final']):.3e} | "
        f"iters L-BFGS = {res['num_iterations']} | "
        f"stop = {res['stop_reason']}"
    )

total_FINAL_time = time.time() - timing_INITIAL
print(f"Temps FINAL total: {total_FINAL_time:.1f}s")
