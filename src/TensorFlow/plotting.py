import numpy as np
from utils import clean_small
import matplotlib.pyplot as plt


def save_run_plots(res, run_dir, L_barre, max_t, N_test, alpha_theo, mode): # Sauvegarde les figures d'un run dans son dossier.
    """
    Sauvegarde les figures d'un run dans son dossier.
    """
    alpha_fit = res["alpha_fit"]

    t_np = np.asarray(res["t_test_phys"]).reshape(-1)
    x_np = np.asarray(res["x_test_phys"]).reshape(-1)
    T_true = res["T_true"]
    T_last = res["T_final_pred"]

    T_fit = np.sin(np.pi * x_np[None, :] / L_barre) * np.exp(-alpha_fit * np.pi**2 * t_np[:, None])
    T_fit[np.abs(T_fit) < 1e-14] = 0.0 # Pour éviter les valeurs très proches de zéro qui pourraient causer des problèmes d'affichage ou de calculs ultérieurs, on utilise la fonction clean_small pour mettre à zéro les éléments de T_fit dont la valeur absolue est inférieure à un seuil tolérable (par exemple, 1e-15). En appliquant clean_small à T_fit, on s'assure que les valeurs très petites sont traitées comme zéro, ce qui peut améliorer la lisibilité des graphiques et éviter les problèmes numériques liés à des valeurs proches de zéro.`

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
        plt.plot(t_np, T_true[:, j], "--", color=color, label=f"théo={alpha_theo:.3f} | x={xv:.2f}")
        plt.plot(t_np, T_fit[:, j], ":", color=color, label=f"fit={alpha_fit:.3f} | x={xv:.2f}")
    plt.xlabel("t")
    plt.ylabel("T")
    plt.title(f"T(t) final (grille fixe) | {mode} | alpha théo vs fit")
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
        plt.plot(x_np, T_true[i, :], "--", color=color, label=f"théo={alpha_theo:.3f} | t={tv:.2f}")
        plt.plot(x_np, T_fit[i, :], ":", color=color, label=f"fit={alpha_fit:.3f} | t={tv:.2f}")
    plt.xlabel("x")
    plt.ylabel("T")
    plt.title(f"T(x) final (grille fixe) | {mode} | alpha théo vs fit")
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
        plt.plot(t_np, T_true[:, j], "--", label=f"théo={alpha_theo:.3f} | x={xv:.2f}")
        plt.plot(t_np, T_fit[:, j], ":", label=f"fit={alpha_fit:.3f} | x={xv:.2f}")
        plt.xlabel("t")
        plt.ylabel("T")
        plt.title(f"T(t) (grille fixe) final x={xv:.2f} | {mode} | alpha théo vs fit")
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
        plt.plot(x_np, T_true[i, :], "--", label=f"théo={alpha_theo:.3f} | t={tv:.2f}")
        plt.plot(x_np, T_fit[i, :], ":", label=f"fit={alpha_fit:.3f} | t={tv:.2f}")
        plt.xlabel("x")
        plt.ylabel("T")
        plt.title(f"T(x) final (grille fixe) t={tv:.2f} | {mode} | alpha théo vs fit")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / f"T_vs_x_final_t_{idx}.png", dpi=200)
        plt.close()
