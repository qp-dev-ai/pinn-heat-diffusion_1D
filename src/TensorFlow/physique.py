 
import jax.numpy as jnp
from jax import grad, vmap

# =========================================================
# SOLUTION EXACTE
# =========================================================
def exact_solution(t_phys, x_phys, alpha_theo, L_tf):
    """
    Solution exacte :
    T(x,t) = exp(-alpha (pi/L)^2 t) sin(pi x / L)
    """
    T = jnp.exp(-alpha_theo * ((jnp.pi / L_tf) ** 2) * t_phys) * jnp.sin((jnp.pi / L_tf) * x_phys)

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


def safe_ic_non_normalized(x, L_tf): # Pour éviter l'effet numérique sin(pi) ~ 1e-16 aux bords, on utilise jnp.where pour forcer la valeur de l'IC à être exactement 0.0 lorsque x est proche de 0.0 ou de L_tf, avec une tolérance de 1e-14. Cela garantit que les conditions initiales sont respectées de manière stricte, ce qui est important pour évaluer correctement la performance du modèle et éviter les erreurs numériques qui pourraient survenir si l'IC prenait des valeurs très petites mais non nulles aux bords.
    ic = jnp.sin((jnp.pi / L_tf) * x)
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
    ic = jnp.sin(jnp.pi * xi)
    ic = jnp.where(jnp.isclose(b, -1.0, atol=1e-14), 0.0, ic)
    ic = jnp.where(jnp.isclose(b,  1.0, atol=1e-14), 0.0, ic)
    return ic

# =========================================================
# HARD IC + HARD BC : ANSATZ CONTRAINT
# =========================================================

def make_model_apply_hard_ic_bc(mode, model_apply_raw, activation_fn, L_tf):
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
        T_raw = model_apply_raw(params, inputs, activation_fn) # Sortie brute du réseau sur les entrées données, qui est utilisée pour calculer la partie non contrainte de la solution proposée par le modèle. En utilisant T_raw, on peut ensuite construire la solution finale T_constrained en ajoutant les parties contraintes (IC et BC) et en multipliant la partie non contrainte par un facteur de disparition qui garantit que les conditions aux bords sont respectées de manière stricte. Cela permet au modèle d'apprendre à satisfaire la PDE tout en respectant exactement les conditions initiales et aux bords imposées par le problème.

        if mode == "non_normalized":
             ic_part = safe_ic_non_normalized(b, L_tf) # Pour la contrainte IC
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
    inp = jnp.array([[a, b]], dtype=jnp.float64) # On crée un tableau d'entrée de shape (1, 2) à partir des coordonnées scalaires a et b, en utilisant jnp.array pour garantir que les données sont au format compatible avec JAX. En utilisant inp, on peut ensuite appeler la fonction model_apply pour obtenir la sortie du modèle sur ce point spécifique, ce qui est nécessaire pour calculer les dérivées nécessaires au calcul du résidu de la PDE et ainsi guider l'entraînement du modèle pour mieux satisfaire la PDE.
    return model_apply(train_params["nn"], inp)[0, 0]

def make_physics_residual(mode, model_apply, get_alpha):
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