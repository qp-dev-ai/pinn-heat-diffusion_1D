## FONCTION UTILES  

import re # Pour la fonction sanitize_filename, qui utilise des expressions régulières pour nettoyer les titres des expériences en supprimant les caractères spéciaux. En utilisant re.sub, on peut facilement remplacer les caractères spéciaux par des underscores ou les supprimer complètement, ce qui permet d'obtenir des titres de fichiers et d'affichage plus propres et plus compatibles avec différents systèmes de fichiers et environnements d'exécution.
import numpy as np


def format_time(seconds): # Formate une durée en secondes en une chaîne lisible (h, m, s)
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
    
    ###### POur mettre les IC à 0 pure
def clean_small(x, tol=1e-15):
    return 0.0 if abs(x) < tol else x

def safe_rel_error(val, ref): # Pour eviter division par 0 dans calcul de l'erreur relative de alpha
    ref = float(ref)
    if abs(ref) < 1e-15:
        return np.nan
    return abs(float(val) - ref) / abs(ref)

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

def make_run_dir(output_root, alpha0, lam,  w, d, s, c, act): # Crée un dossier de run avec un nom basé sur les paramètres d'entraînement, en utilisant make_run_dir pour construire un chemin de dossier unique pour chaque expérience en fonction des paramètres d'entraînement tels que alpha0, lambda_data, width_nn, depth_nn, seed, nombre de points de coloc, et type d'activation. En utilisant make_run_dir, on peut organiser les résultats de chaque expérience dans des dossiers séparés avec des noms descriptifs, ce qui facilite la gestion et l'analyse des résultats obtenus pour différentes configurations d'entraînement.
    run_name = (
        f"alpha0-{alpha0}_lamb-{lam}_w-{w}_d-{d}_seed-{s}_pts_coloc-{c}_act-{act}"
    )
    run_name = sanitize_filename(run_name)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir