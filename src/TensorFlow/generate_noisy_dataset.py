######## GENERE DATA SET BRUITE

import numpy as np
import pandas as pd


def noisy_data_set(dataset_seed, max_t, L_barre, N_data_pts, noise_level, alpha):
    """
    Génère un dataset bruité pour l'équation de la chaleur 1D.
    Supprime les points pour lesquels T_obs < 0.
    """

    # Seed pour reproductibilité
    np.random.seed(dataset_seed)

    # Nombre de points demandé au départ
    N_initial = N_data_pts

    # Points aléatoires
    t = np.random.uniform(0, max_t, N_initial)
    x = np.random.uniform(0, L_barre, N_initial)

    # Solution exacte
    T_clean = np.sin(np.pi * x / L_barre) * np.exp(
        -alpha * (np.pi / L_barre) ** 2 * t
    )

    # Bruit additif gaussien
    sigma = noise_level * np.std(T_clean)
    noise = sigma * np.random.randn(N_initial)

    # Température observée brute
    T_obs_raw = T_clean + noise
    # T_obs_raw = np.clip(T_obs, 0, None) # On s'assure que les températures observées restent positives, car dans le contexte de l'équation de la chaleur, les températures négatives n'ont pas de sens physique. En utilisant np.clip avec une valeur minimale de 0, on remplace toutes les valeurs négatives de T_obs par 0, ce qui garantit que les données observées respectent les contraintes physiques du problème et évite les problèmes liés à des températures négatives dans les étapes ultérieures de l'entraînement du modèle ou de l'analyse des résultats.

    # # Suppression des valeurs négatives
    mask = T_obs_raw >= 0.0

    # TEST DATA SET NORMAL SANS SUPRERESION DES POINTS
    # mask = T_obs_raw >= - 1000.0

    N_final = np.sum(mask)
    N_removed = N_initial - N_final

    kept_ratio = N_final / N_initial
    removed_ratio = N_removed / N_initial

    # Sécurité si tout est supprimé
    if N_final == 0:
        raise ValueError(
            "Tous les points ont été supprimés après filtrage T_obs < 0."
        )

    # Application du masque
    t = t[mask]
    x = x[mask]
    T_clean = T_clean[mask]
    noise = noise[mask]
    T_obs = T_obs_raw[mask]

    # DataFrame
    df = pd.DataFrame({
        "t": t,
        "x": x,
        "T_clean": T_clean,
        "noise": noise,
        "T_obs": T_obs,
        "N_initial": N_initial,
        "N_final": N_final,
        "N_removed": N_removed,
        "kept_ratio": kept_ratio,
        "removed_ratio": removed_ratio,
    })

    print(
        f"Points restants : {N_final}/{N_initial} ({100 * kept_ratio:.1f} %) | "
        f"Points supprimés : {N_removed}/{N_initial} ({100 * removed_ratio:.1f} %)"
    )

    return df, N_final



