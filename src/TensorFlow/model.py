import numpy as np
import jax
import jax.numpy as jnp


## FONCTION D'ACTIVATION POUR LE RESEAU DE NEURONES
def get_activation(name):
    if name == "tanh":
        return jnp.tanh # jnp car fonction universelle
    elif name == "swish": # jax.nn car fonction spécialisée dans jax.nn
        return jax.nn.swish
    else:
        raise ValueError(f"Activation inconnue: {name}")
    
    # =========================================================

# MODELE BRUT
# =========================================================
def glorot_init(key, in_dim, out_dim, dtype=jnp.float64):
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
        W, b = glorot_init(key, layer_dims[k], layer_dims[k + 1], dtype=jnp.float64)
        params.append({"W": W, "b": b})

    return params

def model_apply_raw(params, inputs, activation_fn):
    """
    Sortie brute du réseau sur un batch d'entrées de shape (N, 2).
    """
    x = inputs
    for layer in params[:-1]:
        x = x @ layer["W"] + layer["b"]
        x = activation_fn(x)

    x = x @ params[-1]["W"] + params[-1]["b"]
    return x

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
alpha_floor = jnp.array(1e-12)   # mets 0.0 si tu veux vraiment autoriser alpha=0

def init_alpha_raw_from_alpha(alpha0):
    alpha0 = jnp.array(alpha0, dtype=jnp.float64)
    return jnp.sqrt(jnp.maximum(alpha0 - alpha_floor, 0.0))

def get_alpha(train_params):
    beta = train_params["alpha_raw"]
    return alpha_floor + beta**2