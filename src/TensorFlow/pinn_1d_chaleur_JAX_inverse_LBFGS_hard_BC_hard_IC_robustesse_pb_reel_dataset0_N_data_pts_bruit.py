import os   # pour gérer les chemins de fichiers et les caches
import time
timing_INITIAL = time.time()
import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import qmc # Pour le Latin Hypercube Sampling (LHS) pour la génération de points de la PDE

import jax
import jax.numpy as jnp
from jax import grad, vmap
from jaxopt import LBFGS
from jax.experimental.compilation_cache import compilation_cache as cc
from jax.flatten_util import ravel_pytree 

from pathlib import Path # pour gérer les chemins de fichiers de manière plus robuste et portable, en utilisant Path pour construire les chemins de fichiers de manière indépendante du système d'exploitation, ce qui permet d'éviter les problèmes liés aux séparateurs de chemins et d'améliorer la portabilité du code entre différents environnements. En utilisant Path, on peut facilement créer des chemins de fichiers pour enregistrer les résultats, les figures, les caches de compilation, etc., de manière organisée et sans se soucier des différences entre les systèmes d'exploitation.
import shutil # pour gérer les fichiers et les répertoires, notamment pour supprimer les caches de compilation JAX entre les runs, ce qui permet de libérer de l'espace disque et d'assurer que les compilations sont refaites à partir de zéro pour chaque run, ce qui peut être utile pour garantir que les résultats sont cohérents et comparables entre les différentes expériences. En utilisant shutil.rmtree, on peut facilement supprimer le répertoire de cache de compilation JAX avant chaque run, ce qui garantit que les compilations sont effectuées à nouveau pour chaque expérience et que les résultats ne sont pas influencés par des compilations précédentes.
import re # pour gérer les chaînes de caractères, notamment pour nettoyer les titres des expériences en supprimant les caractères spéciaux qui pourraient poser problème lors de l'enregistrement des fichiers ou de l'affichage des titres. En utilisant re.sub, on peut facilement remplacer les caractères spéciaux par des underscores ou les supprimer complètement, ce qui permet d'obtenir des titres de fichiers et d'affichage plus propres et plus compatibles avec différents systèmes de fichiers et environnements d'exécution.
from collections import deque # pour gérer les files d'attente, notamment pour stocker les métriques d'entraînement au cours des différentes expériences de manière organisée et facilement accessible. En utilisant deque, on peut facilement ajouter de nouvelles métriques à la file d'attente et accéder aux métriques précédentes pour les afficher ou les enregistrer, ce qui facilite le suivi de l'évolution des métriques au fil du temps pendant l'entraînement du modèle.

from openpyxl import Workbook  # Pour mettre en forme fichier excell avec les résultats de toutes les expériences de robustesse, en utilisant openpyxl pour créer un fichier Excel et y enregistrer les résultats de chaque expérience de manière organisée et facilement accessible. En utilisant Workbook, on peut créer des feuilles de calcul pour chaque expérience, ajouter des titres et des données, et appliquer des styles pour améliorer la lisibilité du fichier Excel, ce qui facilite l'analyse et la comparaison des résultats obtenus pour différentes configurations d'entraînement.
from openpyxl.styles import PatternFill, Font
from openpyxl.utils.dataframe import dataframe_to_rows

from tqdm.auto import tqdm # Pour generer barre de progres

def format_time(seconds):
    seconds = int(max(0, seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60

    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m > 0:
        return f"{m}m {s:02d}s"
    else:
        return f"{s}s"


cc.set_cache_dir("./jax_cache") # PERMET DE GARDER EN CACHE LES COMPILATIONS JAX ENTRE LES RUNS, CE QUI ACCELERE BEAUCOUP LES EXÉCUTIONS SUIVANTES APRÈS LA PREMIÈRE.

mpl.rcParams["figure.max_open_warning"] = 100

# =========================================================
# CHOIX DU TYPE NUMERIQUE
# =========================================================
jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64

########################### PARAMETRES TEST ROBUSTESSE ###########################
# Nombre maximal d'itérations L-BFGS
N_epoch = 5000 #default 5000
# Choisir un nombre plus petit de points

parser = argparse.ArgumentParser(description="Génération dataset PINN bruité")

parser.add_argument("--alpha_theo", type=float, default=0.18,
                    help="Valeur théorique de alpha")
parser.add_argument("--noise_level", type=float, default=0.05,
                    help="Niveau de bruit (fraction de std du signal)")
parser.add_argument("--N_data_points", type=int, default=500,
                    help="Nombre de points")
parser.add_argument("--dataset_seed", type=int, default=0,
                    help="Seed pour génération du dataset bruité")
# parser.add_argument("--seed", type=int, default=0,
#                   help="Seed pour reproductibilité")

args = parser.parse_args()

# =========================================================
# PARAMETRES
# =========================================================

alpha_theo = args.alpha_theo
noise_level = args.noise_level # Pourcentage de la variabilité par rapport à la moyenne de T_data (ex 0.05 = 5% de sigma(T))
N_data_points = args.N_data_points

max_t = 1.5 # 1.5 pour consistence avec le dataset
L_barre = 1.0

# alpha_theo = 0.18  # A changer pour generer le dataset

lambdas_data = [0.1, 1, 10] # Differents lambda data dans la loss
widths_nn = [8, 16] # Nombre de neurones
depths_nn = [3] # Profondeur du réseau
lhs_seeds = [0] # Différentes seed pour le LHS (attention n'affecte pas l'initialisation des poids)
colocs = [500] # Nombres de pts de coloc différents pour la PDE
alpha_inits = [0.05, 1.0]

f_activation = "tanh" # Default tanh. 

total_runs = (
    len(alpha_inits)
    * len(lambdas_data)
    * len(widths_nn)
    * len(depths_nn)
    * len(lhs_seeds)
    * len(colocs)
)

print(f"Nombre total de runs = {total_runs}")

######## GENERE DATA SET BRUITE

#np.random.seed(0) # Pour reproductibilité de la génération du dataset, on fixe la seed de numpy à 0 avant de générer les points de données et le bruit, ce qui garantit que les mêmes points et le même bruit sont générés à chaque exécution du code. En utilisant np.random.seed(0), on peut s'assurer que les résultats obtenus pour l'entraînement du modèle sont cohérents et comparables entre différentes runs, ce qui est important pour évaluer la robustesse du modèle face au bruit dans les données.

# Domaine
np.random.seed(args.dataset_seed) # Pour reproductibilité de la génération du dataset, on fixe la seed de numpy à args.dataset_seed avant de générer les points de données et le bruit, ce qui garantit que les mêmes points et le même bruit sont générés à chaque exécution du code.
t = np.random.uniform(0, max_t, N_data_points)
x = np.random.uniform(0, L_barre, N_data_points)

# Solution exacte
T_clean = np.sin(np.pi * x) * np.exp(-alpha_theo * (np.pi**2) * t)

# Bruit gaussien additif
# sigma_bruit = noise_level × std(T_clean)
noise = noise_level * np.std(T_clean) * np.random.randn(N_data_points)
T_obs = T_clean + noise

# Création dataset
df = pd.DataFrame({
    "t": t,
    "x": x,
    "T_clean": T_clean,
    "T_obs": T_obs
})

# Sauvegarde
filename = f"heat1d_inverse_noisy_dataset_{N_data_points}_pts_alpha_{alpha_theo}_noise_level_{noise_level}_set_seed_{args.dataset_seed}.csv"
df.to_csv(filename, index=False)

print(f"Dataset sauvegardé dans : {filename}")


# =========================================================
# CRITERES D'ARRET ANTICIPE
# =========================================================
WINDOW_STOP = 100 # Nombre d'itérations consécutives pendant lesquelles on vérifie les critères d'arrêt anticipé (100 ici). En utilisant WINDOW_STOP, on peut s'assurer que les critères d'arrêt sont vérifiés de manière stable sur une fenêtre d'itérations, ce qui permet d'éviter les arrêts prématurés dus à des fluctuations temporaires des métriques pendant l'entraînement du modèle. En vérifiant les critères d'arrêt sur une fenêtre de plusieurs itérations, on peut garantir que le modèle a suffisamment de temps pour converger vers une solution optimale avant de décider d'arrêter l'entraînement, ce qui peut améliorer la performance finale du modèle et sa capacité à satisfaire la PDE de manière précise.

THRESH_MSE_SOLUTION = 5e-6 
THRESH_MSE_PHYS = 5e-4
THRESH_MSE_DATA = 5e-4
THRESH_MSE_IC = 1e-16
THRESH_MSE_BC = 1e-16

dataset_path = Path(filename)
dataset_name = dataset_path.stem  # -> "heat1d_inverse_noisy_dataset_{N_data_points}pts_alpha_{alpha_theo}_noise_level_{noise_level}"

# Dossier racine de sortie
OUTPUT_ROOT = Path(f"results_robustesse_pb_reel_HARD_IC_BC_{dataset_name}_{f_activation}")

# =========================================================
# PARAMETRES GLOBAUX
# =========================================================
# max_t = 1.5 # 1.5 pour consistence avec le dataset
# L_barre = 1.0
# alpha_val = 0.1
# alpha_raw_init = 10.0

# lr_w = 0.01 # Learning rate pour poids W,b. Default 0.001 (meilleur), increase learning rate to speed up convergence, but can cause instability if too high. Decrease learning rate to 0.00001 for more stable training and better convergence towards la solution exacte, especially in the later stages of training when the model is close to the optimal solution. A lower learning rate allows for finer adjustments of the model weights, which can help to reduce oscillations around the optimal solution and improve the final accuracy of the model. However, it may also increase the time required for training, so it's important to find a balance between convergence speed and stability.
# lr_alpha = 0.1 # Learning rate pour alpha, Default 0.001 (meilleur), increase learning rate to speed up convergence, but can cause instability if too high. Decrease learning rate to 0.00001 for more stable training and better convergence towards la solution exacte, especially in the later stages of training when the model is close to the optimal solution. A lower learning rate allows for finer adjustments of alpha, which can help to reduce oscillations around the optimal solution and improve the final accuracy of the model. However, it may also increase the time required for training, so it's important to find a balance between convergence speed and stability.

# width_nn = 16  # Default 32
# depth_nn = 2    # Default 3


# Nombre de points de colocations fixes pour la PDE
# N_f = 1000 # N_f = 1000 pour comparaison avec TF

# On garde ces paramètres pour monitoring / vérification seulement
N_ic = 100 # Juste pour verifier que les conditions initiales fortes sont bien respectées, pas besoin de beaucoup de points. 
N_bc = 100 # Juste pour verifier que les conditions aux bords fortes sont bien respectées, pas besoin de beaucoup de points. 

# Avec hard IC + hard BC, la loss totale est PDE seule
# lambda_ic = 0.0
# lambda_bc = 0.0
lambda_phys = 1.0
#lambda_data = 100 # = 100 pour comparaison avec TF

# Grille de test
N_test = 50

# Fréquence d'enregistrement / affichage
snap_every = max(1, N_epoch // 10)
append_snap = 1
# append_snap_common = max(1, snap_every // 10)
append_snap_common = 1

# Test final aléatoire
N_rand_final = 10000

pi = jnp.array(np.pi, dtype=DTYPE)
L_tf = jnp.array(L_barre, dtype=DTYPE)
tmax_tf = jnp.array(max_t, dtype=DTYPE)

# Fo = alpha_val * max_t / (L_barre ** 2)
# Fo_tf = jnp.array(Fo, dtype=DTYPE)

print(f"temps max de simulation max_t = {max_t}")
print(f"longueur de la barre L = {L_barre}")
print(f"diffusivité thermique initial alpha = {alpha_inits}")
# print(f"Fourier number Fo = alpha * t_max / L^2 = {Fo:.6f}")

# =========================================================
# IMPORT DATA ET CALCUL DE ALPHA AVEC OPTIMISATION CLASSIQUE (pour comparaison avec la méthode PINN)
# =========================================================

df = pd.read_csv(filename) # 500 points (x, t) tirés aléatoirement, avec la température observée T_obs correspondante, et une diffusivité thermique alpha = 0.18 utilisée pour générer les données. En utilisant pd.read_csv, on peut facilement charger ces données dans un DataFrame pandas, ce qui permet de les manipuler et de les analyser de manière efficace pour calculer la diffusivité thermique alpha à l'aide d'une optimisation classique, en comparant les prédictions du modèle analytique avec les observations du dataset et en ajustant alpha pour minimiser l'erreur entre les deux.
print(f"Dataset original : {len(df)} points")

x_data = df["x"].values
t_data = df["t"].values
T_obs = df["T_obs"].values

x_data_jax = jnp.array(x_data[:, None], dtype=DTYPE)
t_data_jax = jnp.array(t_data[:, None], dtype=DTYPE)
T_obs_jax  = jnp.array(T_obs[:, None], dtype=DTYPE)

# tx_data = jnp.concatenate([t_data_jax, x_data_jax], axis=1)

def residual(alpha):
    a = alpha[0]
    T_analy_data = np.sin(np.pi * x_data) * np.exp(-a * np.pi**2 * t_data)
    return T_analy_data - T_obs

res = least_squares(residual, x0=[0.2], bounds=(0, np.inf)) # x0 (array) est la valeur initiale de alpha pour l'optimisation, et bounds=(0, np.inf) impose que alpha doit être positif pendant l'optimisation, ce qui est important pour garantir que le modèle puisse apprendre à satisfaire la PDE de manière précise. En utilisant least_squares, on cherche à minimiser la somme des carrés des résidus entre les prédictions du modèle et les observations, ce qui permet de trouver la valeur optimale de alpha qui correspond le mieux aux données observées.
alpha_fit = res.x[0]

print("alpha_theorique =", alpha_theo) # VALEUR THEORIQUE DU DATA SET ALPHA = 0.18
print("alpha_fit =", alpha_fit) # VALEUR THEORIQUE DU DATA SET ALPHA = 0.18

###### POur mettre les IC à 0 pure
def clean_small(x, tol=1e-16):
    return 0.0 if abs(x) < tol else x

def safe_rel_error(val, ref): # Pour eviter division par 0 dans calcul RE de alpha
    ref = float(ref)
    if abs(ref) < 1e-15:
        return np.nan
    return abs(float(val) - ref) / abs(ref)

# =========================================================
# ACTIVATION
# =========================================================
def get_activation(name):
    if name == "tanh":
        return jnp.tanh # jnp car fonction universelle
    elif name == "swish": # jax.nn car fonction spécialisée dans jax.nn
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

def make_model_params(width=32, depth=4): # Les nombres sont arbitraires ici, ils seront appelés plus tard
    layer_dims = [2] + [width] * depth + [1]
     
    params = []
    for k in range(len(layer_dims) - 1):
        seed = 100 + k if k < depth else 200
        key = jax.random.PRNGKey(seed) # Pour garantir des poids initiaux identiques entre les différentes expériences, on fixe la seed en fonction de la couche k. Les couches du réseau (k < depth) ont des seeds différentes (100, 101, 102, ...) pour assurer une diversité dans l'initialisation des poids, tandis que la dernière couche (k = depth) a une seed différente (200) pour garantir une initialisation distincte de ses poids. En utilisant jax.random.PRNGKey avec ces seeds spécifiques, on s'assure que les poids initiaux du modèle sont les mêmes à chaque exécution du code, ce qui permet une comparaison équitable entre les différentes stratégies d'entraînement et d'optimisation utilisées dans les expériences.
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
    T = jnp.exp(-alpha_theo * ((pi / L_tf) ** 2) * t_phys) * jnp.sin((pi / L_tf) * x_phys)

    # Pour éviter l'effet numerique sin(pi) ~ 1e-16 aux bords
    T = jnp.where(jnp.isclose(x_phys, 0.0, atol=1e-14), 0.0, T) # Pour éviter l'effet numérique sin(pi) ~ 1e-16 aux bords, on utilise jnp.where pour forcer la valeur de T à être exactement 0.0 lorsque x_phys est proche de 0.0 ou de L_tf, avec une tolérance de 1e-14. Cela garantit que les conditions aux bords sont respectées de manière stricte, ce qui est important pour évaluer correctement la performance du modèle et éviter les erreurs numériques qui pourraient survenir si T prenait des valeurs très petites mais non nulles aux bords.
    T = jnp.where(jnp.isclose(x_phys, L_tf, atol=1e-14), 0.0, T)
    return T

# =========================================================
# STRATEGIES DE NORMALISATION (Pour le probleme inverse pas de normalisation qui dépend de alpha, car on veut que le modele puisse apprendre a ajuster alpha pour mieux satisfaire la PDE, et une normalisation qui depend de alpha pourrait rendre l'apprentissage plus difficile en introduisant une dépendance circulaire entre alpha et la normalisation. En utilisant des coordonnées physiques non normalisées, on permet au modèle d'apprendre à ajuster alpha de manière plus flexible pour minimiser le résidu de la PDE, ce qui peut conduire à une meilleure convergence vers la solution exacte.)
# =========================================================
def phys_to_non_normalized(t_phys, x_phys):
    return t_phys, x_phys

# def phys_to_nd_only(t_phys, x_phys):
#     tau = alpha * t_phys / (L_tf ** 2)
#     xi = x_phys / L_tf
#     return tau, xi

# def phys_to_nd_centered(t_phys, x_phys): # CENTRAGE ET MISE A L'ECHELLE uniquement (sans normalisation)
#     t_net = 2.0 * t_phys / tmax_tf - 1.0
#     x_net = 2.0 * x_phys / L_tf - 1.0
#     return t_net, x_net


def safe_ic_non_normalized(x): # Pour éviter l'effet numérique sin(pi) ~ 1e-16 aux bords, on utilise jnp.where pour forcer la valeur de l'IC à être exactement 0.0 lorsque x est proche de 0.0 ou de L_tf, avec une tolérance de 1e-14. Cela garantit que les conditions initiales sont respectées de manière stricte, ce qui est important pour évaluer correctement la performance du modèle et éviter les erreurs numériques qui pourraient survenir si l'IC prenait des valeurs très petites mais non nulles aux bords.
    ic = jnp.sin((pi / L_tf) * x)
    ic = jnp.where(jnp.isclose(x, 0.0, atol=1e-14), 0.0, ic) # Pour éviter l'effet numérique sin(pi) ~ 1e-16 aux bords, on utilise jnp.where pour forcer la valeur de l'IC à être exactement 0.0 lorsque x est proche de 0.0 ou de L_tf, avec une tolérance de 1e-14. Cela garantit que les conditions initiales sont respectées de manière stricte, ce qui est important pour évaluer correctement la performance du modèle et éviter les erreurs numériques qui pourraient survenir si l'IC prenait des valeurs très petites mais non nulles aux bords.
    ic = jnp.where(jnp.isclose(x, L_tf, atol=1e-14), 0.0, ic)
    return ic

# def safe_ic_nd_only(xi):
#     ic = jnp.sin(pi * xi)
#     ic = jnp.where(jnp.isclose(xi, 0.0, atol=1e-14), 0.0, ic)
#     ic = jnp.where(jnp.isclose(xi, 1.0, atol=1e-14), 0.0, ic)
#     return ic

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
        T_raw = model_apply_raw(params, inputs) # Sortie brute du réseau sur les entrées données, qui est utilisée pour calculer la partie non contrainte de la solution proposée par le modèle. En utilisant T_raw, on peut ensuite construire la solution finale T_constrained en ajoutant les parties contraintes (IC et BC) et en multipliant la partie non contrainte par un facteur de disparition qui garantit que les conditions aux bords sont respectées de manière stricte. Cela permet au modèle d'apprendre à satisfaire la PDE tout en respectant exactement les conditions initiales et aux bords imposées par le problème.

        if mode == "non_normalized":
             ic_part = safe_ic_non_normalized(b) # Pour la contrainte IC
             vanish_factor = a * b * (L_tf - b) # Pour la contrainte BC, on utilise un facteur de disparition vanish_factor qui est égal à a * b * (L_tf - b). Ce facteur garantit que la partie non contrainte T_raw est multipliée par zéro lorsque b est égal à 0 ou à L_tf, ce qui impose que T(t,0) = 0 et T(t,L) = 0 pour tous les temps t. De plus, le facteur a garantit que la partie non contrainte disparaît également à t=0, ce qui permet de respecter la condition initiale T(0,x) = sin(pi x / L) de manière stricte. En combinant ces deux facteurs de

        # elif mode == "nd_only":
        #      ic_part = safe_ic_nd_only(b)
        #      vanish_factor = a * b * (1.0 - b)

        # elif mode == "nd_centered":
        #      ic_part = safe_ic_centered(b)
        #      vanish_factor = (a + 1.0) * (1.0 - b**2)

        else:
            raise ValueError("Mode inconnu")

        return ic_part + vanish_factor * T_raw

    return model_apply_constrained

# =========================================================
# RESIDUS PDE SELON LA STRATEGIE
# =========================================================
def scalar_model_output(train_params, a, b, model_apply): # Retourne la temperature predite sur un seul point
    """
    Sortie scalaire du modèle contraint sur un seul point (a, b).
    """
    inp = jnp.array([[a, b]], dtype=DTYPE)
    return model_apply(train_params["nn"], inp)[0, 0]

def make_physics_residual(mode, model_apply):
    """
    Résidu natif PDE calculé sur la sortie CONTRAINTE.
    """
    def residual_single(train_params, a, b):
        dT_da = grad(scalar_model_output, argnums=1)(train_params, a, b, model_apply)
        d2T_db2 = grad(grad(scalar_model_output, argnums=2), argnums=2)(train_params, a, b, model_apply) # avec scalar_model_output, les poids que entraines que sur le NN

        if mode == "non_normalized":
            # a=t, b=x
            alpha = get_alpha(train_params)  # Récupère la valeur de alpha à une iteration donnée de l'optimisation, en utilisant la fonction get_alpha qui extrait la valeur de alpha à partir des paramètres du modèle train_params. En utilisant get_alpha(train_params), on peut ensuite calculer le résidu de la PDE en utilisant la valeur actuelle de alpha, ce qui permet au modèle d'apprendre à ajuster alpha pour minimiser ce résidu et ainsi mieux satisfaire la PDE.
            r = dT_da - alpha * d2T_db2

        # elif mode == "nd_only":
        #     # a=tau, b=xi
        #     r = dT_da - d2T_db2

        # elif mode == "nd_centered":
        #     # a=t_net, b=x_net
        #     alpha = get_alpha(train_params)
        #     r = dT_da - (2.0 * alpha * tmax_tf / (L_tf ** 2)) * d2T_db2


        else:
            raise ValueError("Mode inconnu")

        return r

    residual_batch = vmap(residual_single, in_axes=(None, 0, 0)) # IMPORTANT. Vectorisation du calcul du résidu PDE sur un batch de points (a,b), en utilisant vmap pour appliquer la fonction residual_single à chaque point du batch. En utilisant vmap, on peut calculer efficacement le résidu de la PDE pour un grand nombre de points en parallèle, ce qui accélère considérablement l'entraînement du modèle et permet d'obtenir des résultats plus rapidement. Le résidu batch r_batch est ensuite utilisé dans la fonction physics_residual pour calculer la perte de la PDE sur un batch de points, ce qui guide l'optimisation du modèle pour mieux satisfaire la PDE.

    def physics_residual(params, tx): # Applique le vmap du résidu PDE sur un batch de points tx, en extrayant les coordonnées a et b de tx et en calculant le résidu pour chaque point à l'aide de residual_batch. En utilisant physics_residual(params, tx), on peut ensuite calculer la perte de la PDE sur un batch de points pendant l'entraînement du modèle, ce qui permet d'optimiser les paramètres du modèle pour minimiser cette perte et ainsi mieux satisfaire la PDE.
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
    # elif mode == "nd_only":
    #     return (alpha / (L_tf ** 2)) * r_native
    # elif mode == "nd_centered":
    #     return (2.0 / tmax_tf) * r_native
    else:
        raise ValueError("Mode inconnu")

# =========================================================
# OUTILS
# =========================================================
def mse(a, b):
    return jnp.mean((a - b) ** 2)
    
######## FONCTION POUR SAUVER LES FIGURES #################

def sanitize_filename(name): # Pour nettoyer les titres des expériences en supprimant les caractères spéciaux qui pourraient poser problème lors de l'enregistrement des fichiers ou de l'affichage des titres. En utilisant re.sub, on peut facilement remplacer les caractères spéciaux par des underscores ou les supprimer complètement, ce qui permet d'obtenir des titres de fichiers et d'affichage plus propres et plus compatibles avec différents systèmes de fichiers et environnements d'exécution.
    """
    Rend une chaîne sûre pour nom de fichier/dossier.
    """
    name = str(name)
    name = name.replace(" ", "_")
    name = name.replace("|", "__")
    name = name.replace("=", "-")
    name = name.replace(".", "p")
    name = re.sub(r"[^A-Za-z0-9_\-]", "", name)
    return name


def make_run_dir(output_root, alpha0, lam, w, d, s, c, act): # Crée un dossier de run avec un nom basé sur les paramètres d'entraînement, en utilisant make_run_dir pour construire un chemin de dossier unique pour chaque expérience en fonction des paramètres d'entraînement tels que alpha0, lambda_data, width_nn, depth_nn, seed, nombre de points de coloc, et type d'activation. En utilisant make_run_dir, on peut organiser les résultats de chaque expérience dans des dossiers séparés avec des noms descriptifs, ce qui facilite la gestion et l'analyse des résultats obtenus pour différentes configurations d'entraînement.
    run_name = (
        f"alpha0-{alpha0}_lamb-{lam}_w-{w}_d-{d}_seed-{s}_pts_coloc-{c}_act-{act}"
    )
    run_name = sanitize_filename(run_name)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_plots(res, run_dir, L_barre, max_t, N_test, alpha_theo, mode): # Sauvegarde les figures d'un run dans son dossier.
    """
    Sauvegarde les figures d'un run dans son dossier.
    """
    t_np = res["t_test_phys"]
    x_np = res["x_test_phys"]
    T_true = res["T_true"]
    T_last = res["T_final_pred"]

    # -------------------------
    # 1) Evolution alpha
    # -------------------------
    plt.figure()
    plt.plot(res["epoch_history"], res["alpha_val_history"], label="alpha appris")
    plt.axhline(alpha_theo, linestyle="--", label="alpha théorique (du dataset)")
    plt.axhline(alpha_fit, linestyle="--", color = "purple", label="alpha fit (du dataset)")
    plt.xlabel("Itérations / appels L-BFGS")
    plt.ylabel("alpha")
    plt.title("Evolution du coeff de diffusion (alpha)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "alpha_evolution.png", dpi=200)
    plt.close()

    # -------------------------
    # 2) Loss totale
    # -------------------------
    plt.figure()
    plt.plot(res["epoch_history"], res["loss_history"])
    plt.xlabel("Itérations / appels L-BFGS")
    plt.ylabel("Total Loss (PDE + data)")
    plt.title("Total loss (sur pts de coloc LHS + pts exp)")
    plt.tight_layout()
    plt.savefig(run_dir / "loss_total.png", dpi=200)
    plt.close()

    # -------------------------
    # 3) Loss physique
    # -------------------------
    plt.figure()
    plt.plot(res["epoch_history"], res["loss_phys_history"])
    plt.xlabel("Itérations / appels L-BFGS")
    plt.ylabel("Physics Loss")
    plt.title("Physics loss (sur pts de coloc - tirage aléatoire fixe LHS)")
    plt.tight_layout()
    plt.savefig(run_dir / "loss_phys.png", dpi=200)
    plt.close()

    # -------------------------
    # 4) Loss data
    # -------------------------
    plt.figure()
    plt.plot(res["epoch_history"], res["loss_data_history"])
    plt.xlabel("Itérations / appels L-BFGS")
    plt.ylabel("Data Loss")
    plt.title("Data loss (calc sur pts exp du dataset)")
    plt.tight_layout()
    plt.savefig(run_dir / "loss_data.png", dpi=200)
    plt.close()

    # -------------------------
    # 5) IC / BC check
    # -------------------------
    plt.figure()
    plt.plot(res["epoch_history"], res["loss_ic_history"])
    plt.xlabel("Itérations / appels L-BFGS")
    plt.ylabel("IC check")
    plt.title("Vérification IC dure (sur pts fixes en espace)")
    plt.tight_layout()
    plt.savefig(run_dir / "ic_check.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(res["epoch_history"], res["loss_bc_history"])
    plt.xlabel("Itérations / appels L-BFGS")
    plt.ylabel("BC check")
    plt.title("Vérification BC dure (sur pts fixes en temps)")
    plt.tight_layout()
    plt.savefig(run_dir / "bc_check.png", dpi=200)
    plt.close()

    # -------------------------
    # 6) MSE solution / phys commune / total commune
    # -------------------------
    plt.figure()
    plt.plot(res["epoch_common_history"], res["mse_solution_history"])
    plt.xlabel("Itérations / appels L-BFGS")
    plt.ylabel("MSE solution")
    plt.title("MSE solution sur grille fixe")
    plt.tight_layout()
    plt.savefig(run_dir / "mse_solution_fixed_grid.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(res["epoch_common_history"], res["loss_phys_common_history"])
    plt.xlabel("Itérations / appels L-BFGS")
    plt.ylabel("Physics Loss")
    plt.title("Physics loss sur grille fixe (pr comp normalisation ou inter run)")
    plt.tight_layout()
    plt.savefig(run_dir / "loss_phys_common.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(res["epoch_common_history"], res["loss_total_common_history"])
    plt.xlabel("Itérations / appels L-BFGS")
    plt.ylabel("Total Loss (PDE + data)")
    plt.title("Total loss (sur grille fixe + pts exp)")
    plt.tight_layout()
    plt.savefig(run_dir / "loss_total_common.png", dpi=200)
    plt.close()

    # -------------------------
    # 7) Courbes T(t) à x fixés
    # -------------------------
    x_vals = np.linspace(0.0, L_barre, 6)
    j_list = [int(round((xv / L_barre) * (N_test - 1))) for xv in x_vals]

    plt.figure()
    for j, xv in zip(j_list, x_vals):
        line_model, = plt.plot(t_np, T_last[:, j], label=f"modèle | x={xv:.2f}")
        color = line_model.get_color()
        plt.plot(t_np, T_true[:, j], "--", color=color, label=f"exact | x={xv:.2f}")
    plt.xlabel("t")
    plt.ylabel("T")
    plt.title(f"T(t) final (grille fixe) | mode = {mode}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "T_vs_t_final.png", dpi=200)
    plt.close()

    # -------------------------
    # 8) Courbes T(x) à t fixés
    # -------------------------
    t_vals = np.linspace(0.0, max_t, 6)
    i_list = [int(round((tv / max_t) * (N_test - 1))) for tv in t_vals]

    plt.figure()
    for i, tv in zip(i_list, t_vals):
        line_model, = plt.plot(x_np, T_last[i, :], label=f"modèle | t={tv:.2f}")
        color = line_model.get_color()
        plt.plot(x_np, T_true[i, :], "--", color=color, label=f"exact | t={tv:.2f}")
    plt.xlabel("x")
    plt.ylabel("T")
    plt.title(f"T(x) final (grille fixe) | mode = {mode}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(run_dir / "T_vs_x_final.png", dpi=200)
    plt.close()

       # -------------------------
    # 9) 6 plots individuels T(t) pour les mêmes positions x
    # -------------------------
    for idx, (j, xv) in enumerate(zip(j_list, x_vals), start=1):
        plt.figure()
        plt.plot(t_np, T_last[:, j], label=f"modèle | x={xv:.2f}")
        plt.plot(t_np, T_true[:, j], "--", label=f"exact | x={xv:.2f}")
        plt.xlabel("t")
        plt.ylabel("T")
        plt.title(f"T(t) (grille fixe) final à x={xv:.2f} | mode = {mode}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / f"T_vs_t_final_x_{idx}.png", dpi=200)
        plt.close()

    # -------------------------
    # 10) 6 plots individuels T(x) pour les mêmes temps t
    # -------------------------
    for idx, (i, tv) in enumerate(zip(i_list, t_vals), start=1):
        plt.figure()
        plt.plot(x_np, T_last[i, :], label=f"modèle | t={tv:.2f}")
        plt.plot(x_np, T_true[i, :], "--", label=f"exact | t={tv:.2f}")
        plt.xlabel("x")
        plt.ylabel("T")
        plt.title(f"T(x) final (grille fixe) à t={tv:.2f} | mode = {mode}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / f"T_vs_x_final_t_{idx}.png", dpi=200)
        plt.close()


#### Des fonctions pour calculer les normes des gradients, qui peuvent être utiles pour monitorer la convergence de l'optimisation et détecter d'éventuels problèmes de vanishing ou exploding gradients pendant l'entraînement du modèle. En utilisant grad_norm(train_params, data), on peut calculer la norme des gradients de la perte par rapport aux paramètres du modèle à chaque itération de l'optimisation, ce qui permet d'avoir une meilleure compréhension de la dynamique de l'entraînement et d'ajuster les hyperparamètres si nécessaire pour améliorer la convergence vers la solution optimale.

# =========================================================
# EXPERIENCE L-BFGS
# =========================================================
def run_experiment(mode, title, base_train_params, lambda_data_var, seed_var, n_coloc, run_dir, run_idx, total_runs):
    print("\n" + "=" * 70)
    progress_pct = 100.0 * run_idx / total_runs
    print(f"EXPERIENCE {run_idx}/{total_runs} ({progress_pct:.1f}%) : {title}")
    print("=" * 70)

    # np.random.seed(0) # plus utile maintenant car seed deja dans le LHS

    train_params = copy.deepcopy(base_train_params)

    if mode == "non_normalized":
        phys_to_net = phys_to_non_normalized
    # elif mode == "nd_only":
    #     phys_to_net = phys_to_nd_only
    # elif mode == "nd_centered":
    #     phys_to_net = phys_to_nd_centered
    else:
        raise ValueError("Mode inconnu")

    # Modèle contraint hard IC + hard BC
    model_apply = make_model_apply_hard_ic_bc(mode)

    # Résidu PDE calculé sur ce modèle contraint
    physics_residual = make_physics_residual(mode, model_apply) # On calcule le résidu apres avoir calculé, pour que le résidu soit calculé sur la sortie contrainte du modèle, ce qui garantit que les conditions initiales et aux bords sont respectées de manière stricte pendant l'entraînement. En utilisant physics_residual, on peut ensuite calculer la perte de la PDE en utilisant ce résidu, ce qui guide l'optimisation du modèle pour mieux satisfaire la PDE tout en respectant les contraintes imposées par les conditions initiales et aux bords.

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

    T_true = exact_solution(Tg_phys, Xg_phys) # Calcul de la solution exacte sur la grille de test, qui est utilisée pour évaluer la performance du modèle en comparant les prédictions du modèle avec cette solution exacte. En utilisant T_true, on peut calculer des métriques d'évaluation telles que le MSE entre les prédictions du modèle et la solution exacte, ce qui permet de mesurer la précision du modèle et sa capacité à apprendre à satisfaire la PDE de manière précise.

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
        T_pred_obs = model_apply(train_params["nn"], data["tx_data_net"]) # Prédictions du modèle pour les points de données observées (t_data,x_data), qui sont utilisées pour calculer la perte de données en comparant T_pred_obs avec T_obs, qui est la température observée dans le dataset. L'objectif est de minimiser cette perte pour que le modèle puisse apprendre à faire des prédictions précises pour les points de données observées, ce qui peut aider à améliorer la précision globale du modèle et sa capacité à satisfaire la PDE.
        loss_data = jnp.mean((T_pred_obs - T_obs_jax)**2)

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
                    f"loss phys (commune, grille): {clean_small(float(metrics['loss_phys_common'])):.3e}"
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
    T_true_final = exact_solution(t_final_phys, x_final_phys)

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

# =========================================================
# POIDS INITIAUX COMMUNS
# =========================================================

# SOFTPLUS POUR ALPHA

# alpha_min = jnp.array(0.0, dtype=DTYPE) # Pour eviter que alpha appris = 0 mais alpha peut apprendre alpha = alpha_min

# def inverse_softplus(y):
#     eps = jnp.array(1e-12, dtype=DTYPE)
#     y = jnp.maximum(y, eps)
#     return jnp.log(jnp.expm1(y))

# ####---OBTENIR ALPHA ENTRAINEMENT
# def get_alpha(train_params):
#     return alpha_min + jax.nn.softplus(train_params["alpha_raw"]) # Force alpha > 0, mais le réseau peut apprendre alpha proche de alpha min

## TEST ALPHA EN SQRT (DYNAMIQUE plus douce que softplus) POUR RALENTIR APPRENTISSAGE DE ALPHA ET EVITER ALPHA =0
alpha_floor = jnp.array(1e-12, dtype=DTYPE)   # mets 0.0 si tu veux vraiment autoriser alpha=0

def init_alpha_raw_from_alpha(alpha0):
    alpha0 = jnp.array(alpha0, dtype=DTYPE)
    return jnp.sqrt(jnp.maximum(alpha0 - alpha_floor, 0.0))

def get_alpha(train_params):
    beta = train_params["alpha_raw"]
    return alpha_floor + beta**2

# =========================================================
# RUN DES 3 EXPERIENCES
# =========================================================

# =========================================================
# PREPARE DOSSIER DE SORTIE
# =========================================================

def save_summary_outputs(all_results, output_root):
    """
    Sauvegarde un CSV + un Excel formaté avec tous les runs déjà terminés.
    Peut être appelée après chaque run pour éviter de perdre les résultats
    en cas d'arrêt manuel ou de plantage.
    """
    if len(all_results) == 0:
        return

    summary_rows = []
    for res in all_results:
        summary_rows.append({
            "title": res["title"],
            "dataset_noise": res["noise_level"],
            "dataset_pts": res["N_data_points"],
            "dataset_seed": res["dataset_seed"],
            "alpha0": res["alpha0"],
            "lambda_data": res["lambda_data"],
            "width": res["width"],
            "depth": res["depth"],
            "lhs_seed": res["lhs_seed"],
            "coloc": res["coloc"],
            "activation": res["activation"],
            "alpha_theo": res["alpha_theo"],
            "alpha_fit": res["alpha_fit"],
            "alpha_final": res["alpha_final"],
            "alpha_RE_fit (%)": round(safe_rel_error(res["alpha_final"], res["alpha_fit"]) * 100, 2),
            "alpha_RE_theo (%)": round(safe_rel_error(res["alpha_final"], res["alpha_theo"]) * 100, 2),
            "mse_final": res["mse_final"],
            "mse_residual_phys": res["mse_residual_phys"],
            "mse_data_final": res["mse_data_final"],
            "mse_ic_final": clean_small(res["mse_ic_final"]),
            "mse_bc_final": clean_small(res["mse_bc_final"]),
           # "criteria_solution_ok": res["criteria_solution_ok"],
            "criteria_phys_ok": res["criteria_phys_ok"],
            "criteria_data_ok": res["criteria_data_ok"],
            "criteria_ic_ok": res["criteria_ic_ok"],
            "criteria_bc_ok": res["criteria_bc_ok"],
            "all_final_mse_ok": res["all_final_mse_ok"],
            "num_iterations": res["num_iterations"],
            "total_loss_finale": res["total_loss_finale"],
            "converged": res["converged"],
            "grad_LBFGS_final": res["grad_LBFGS_final"],
            "failed (Nan ou Inf)": res["failed (Nan ou Inf)"],
            "stop_reason": res["stop_reason"],
            "run_dir": res["run_dir"],
        })

    df_summary = pd.DataFrame(summary_rows)

    # Sauvegarde CSV immédiate
    csv_path = output_root / f"summary_all_runs_HARD_IC_BC_{dataset_name}_{f_activation}.csv"
    df_summary.to_csv(csv_path, index=False)

    # Sauvegarde Excel formatée
    xlsx_path = output_root / f"summary_all_runs_HARD_{dataset_name}_{f_activation}.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = "summary"

    for row in dataframe_to_rows(df_summary, index=False, header=True):
        ws.append(row)

    fill_green = PatternFill(fill_type="solid", start_color="C6EFCE", end_color="C6EFCE")
    fill_yellow = PatternFill(fill_type="solid", start_color="FFF2CC", end_color="FFF2CC")
    fill_red = PatternFill(fill_type="solid", start_color="F4CCCC", end_color="F4CCCC")
    font_bad_mse = Font(color="C00000", bold=True)

    header = [cell.value for cell in ws[1]]
    col_idx = {name: i + 1 for i, name in enumerate(header)}

    threshold_map = {
    #    "mse_final": THRESH_MSE_SOLUTION,
        "mse_residual_phys": THRESH_MSE_PHYS,
        "mse_data_final": THRESH_MSE_DATA,
        "mse_ic_final": THRESH_MSE_IC,
        "mse_bc_final": THRESH_MSE_BC,
    }

    for row in range(2, ws.max_row + 1):
        all_mse_ok = True

        for col_name, thresh in threshold_map.items():
            val = ws.cell(row=row, column=col_idx[col_name]).value

            if val is None or val >= thresh:
                all_mse_ok = False

            if val is not None and val >= thresh:
                ws.cell(row=row, column=col_idx[col_name]).font = font_bad_mse

        stop_reason_val = ws.cell(row=row, column=col_idx["stop_reason"]).value

        if all_mse_ok:
            row_fill = fill_green
        elif stop_reason_val == "lbfgs_converged":
            row_fill = fill_yellow
        else:
            row_fill = fill_red

        for col in range(1, ws.max_column + 1):
            ws.cell(row=row, column=col).fill = row_fill

    for col in ws.columns:
        max_length = 0
        col_letter = col[0].column_letter
        for cell in col:
            try:
                max_length = max(max_length, len(str(cell.value)))
            except:
                pass
        ws.column_dimensions[col_letter].width = min(max_length + 2, 40)

    wb.save(xlsx_path)

if OUTPUT_ROOT.exists(): # Si le dossier de sortie existe déjà, on le supprime pour éviter d'avoir des résultats mélangés avec les anciennes expériences. En utilisant shutil.rmtree(OUTPUT_ROOT), on peut supprimer de manière récursive le dossier de sortie et tout son contenu, ce qui garantit que les résultats des nouvelles expériences seront stockés dans un dossier propre et organisé. Ensuite, en utilisant OUTPUT_ROOT.mkdir(parents=True, exist_ok=True), on recrée le dossier de sortie pour pouvoir y enregistrer les résultats des nouvelles expériences.
    shutil.rmtree(OUTPUT_ROOT)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

all_results = [] # Chaque élément de all_results contient toutes les données pour une expérience donc all_results[0] = run 1, all_results[1] = run 2, etc... et chaque run contient les losses, les snapshots, les métriques d'évaluation, etc... pour cette expérience spécifique. En utilisant all_results, on peut ensuite faire des comparaisons entre les différentes expériences en analysant les données contenues dans chaque élément de la liste, ce qui permet d'obtenir des insights sur l'impact de chaque stratégie d'entraînement et d'optimisation utilisée dans les expériences.
run_counter = 0
timing_global_start = time.time()

pbar = tqdm(total=total_runs, desc="Progression globale", unit="run", position=0)

try:
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
                                "alpha_raw": init_alpha_raw_from_alpha(alpha0).astype(DTYPE),
                            }
                                                        
                            title = f"alpha0={alpha0} | lam={lam} | w={w} | d={d} | lhs_seed={s} | coloc={c} | act={f_activation}"

                            run_dir = make_run_dir(
                                OUTPUT_ROOT,
                                alpha0=alpha0,
                                lam=lam,
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
                            )

                            results["alpha0"] = alpha0
                            results["alpha_fit"] = alpha_fit
                            results["alpha_theo"] = alpha_theo
                            results["noise_level"] = noise_level
                            results["N_data_points"] = N_data_points
                            results["dataset_seed"] = args.dataset_seed
                            results["width"] = w
                            results["depth"] = d
                            results["lhs_seed"] = s
                            results["coloc"] = c
                            results["activation"] = f_activation

                            all_results.append(results)

                            # CHECKPOINT APRES CHAQUE RUN
                            save_summary_outputs(all_results, OUTPUT_ROOT)

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
    save_summary_outputs(all_results, OUTPUT_ROOT)

except Exception as e:
    print(f"\nErreur globale détectée : {e}")
    save_summary_outputs(all_results, OUTPUT_ROOT)
    raise

finally:
    pbar.close()

save_summary_outputs(all_results, OUTPUT_ROOT)

# results_nd  = run_experiment("nd_only", "2) Nondimensionnalisation physique seule", base_train_params)
# results_ndc = run_experiment("nd_centered", "2) centrage [-1,1]", base_train_params)


#all_results = [results_non, results_ndc]

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
        f"MSE data check = {res['mse_data_final']:.3e} | "
        f"MSE IC check = {clean_small(res['mse_ic_final']):.3e} | "
        f"MSE BC check = {clean_small(res['mse_bc_final']):.3e} | "
        f"iters L-BFGS = {res['num_iterations']} | "
        f"stop = {res['stop_reason']}"
    )

total_FINAL_time = time.time() - timing_INITIAL
print(f"Temps FINAL total: {total_FINAL_time:.1f}s")

plt.close("all")