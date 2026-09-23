import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font
from openpyxl.utils.dataframe import dataframe_to_rows

from utils import safe_rel_error, clean_small


# =========================================================
# PREPARE DOSSIER DE SORTIE 
# =========================================================

# Dans le fichier excell les lignes vertes respectent le critère MSE DATA ET PHYSIQUE (pas de MSE solution en pb reel)
# Les lignes rouges ne respectent pas le critère MSE data ou physique
# Les lignes jaunes ont convergé d'après L-BFGS mais n'atteignent pas les critères de MSE data ou physique (ex: convergé vers un minimum local non satisfaisant)

def save_summary_outputs(
    all_results,
    output_root,
    dataset_name,
    f_activation,
    loss_mode,
    thresh_mse_phys,
    thresh_mse_data,
    thresh_mse_ic,
    thresh_mse_bc,                                 
):
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
            "dataset_pts_initial": res["N_data_points_initial"],
            "dataset_pts_final": res["N_data_points_final"],
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
            "thresh_mse_phys": res["thresh_mse_phys"],
            "phys_over_thresh": res["mse_residual_phys"] / (res["thresh_mse_phys"] + 1e-12),
            "mse_data_final": res["mse_data_final"],
            "thresh_mse_data": res["thresh_mse_data"],
            "mse_data_over_thresh": res["mse_data_final"] / (res["thresh_mse_data"] + 1e-12),
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

    if dataset_name == "GLOBAL":
        xlsx_path = output_root / f"summary_GLOBAL_{f_activation}_{loss_mode}.xlsx"
        csv_path = output_root / f"summary_GLOBAL_{f_activation}_{loss_mode}.csv"

    else:
        xlsx_path = output_root / f"results.xlsx"
        csv_path = output_root / f"results.csv"
        
    df_summary.to_csv(csv_path, index=False)
    
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

    for row in range(2, ws.max_row + 1):
        all_mse_ok = True

        # Physique : seuil variable par run
        val_phys = ws.cell(row=row, column=col_idx["mse_residual_phys"]).value
        thresh_phys_row = ws.cell(row=row, column=col_idx["thresh_mse_phys"]).value

        if val_phys is None or thresh_phys_row is None or val_phys >= thresh_phys_row:
            all_mse_ok = False

        if val_phys is not None and thresh_phys_row is not None and val_phys >= thresh_phys_row:
            ws.cell(row=row, column=col_idx["mse_residual_phys"]).font = font_bad_mse


        # IC/BC : seuils fixes
        fixed_thresholds = {
            "mse_ic_final": thresh_mse_ic,
            "mse_bc_final": thresh_mse_bc,
        }

        for col_name, thresh in fixed_thresholds.items():
            val = ws.cell(row=row, column=col_idx[col_name]).value

            if val is None or val >= thresh:
                all_mse_ok = False

            if val is not None and val >= thresh:
                ws.cell(row=row, column=col_idx[col_name]).font = font_bad_mse

        val_data = ws.cell(row=row, column=col_idx["mse_data_final"]).value
        thresh_data_row = ws.cell(row=row, column=col_idx["thresh_mse_data"]).value

        if val_data is None or thresh_data_row is None or val_data >= thresh_data_row:
            all_mse_ok = False

        if val_data is not None and thresh_data_row is not None and val_data >= thresh_data_row:
            ws.cell(row=row, column=col_idx["mse_data_final"]).font = font_bad_mse

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