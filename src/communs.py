import numpy as np
import pandas as pd
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time



def load_data(filepath):
    timestep= 15 #minutes
    rows_per_day = int(24 * 60 / timestep)  # 96
    design_days =2*rows_per_day+1
    dict_newindex = {
        'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)': 'Tout',
        'LIVING_UNIT1:Zone Air Temperature [C](TimeStep)': 'Tzone',
        'LIVING_UNIT1:Zone Air System Sensible Heating Rate [W](TimeStep)': 'Heating_living_unit',
        'LIVING_UNIT1:Zone Air System Sensible Cooling Rate [W](TimeStep)': 'Cooling_living_unit',
    }
    df = pd.read_csv(filepath, sep=";", skiprows=range (1, design_days))
    #print(df.columns)
    df.columns = df.columns.str.strip()

    df.rename(columns=dict_newindex, inplace=True)

    # décalage temporel
    df['Tzone_next'] = df['Tzone'].shift(-1)
    df['Qhvac'] = df['Heating_living_unit'] - df['Cooling_living_unit']
    # Suppression de la dernière ligne (NaN)
    df = df.dropna()


    # Séparer la première colonne en plusieurs colonnes selon un délimiteur
    #df_propre = d.iloc[:, 0].str.split(',', expand=True)

    # décalage temporel
    # df["Tzone_next"] = df["Tzone"].shift(-1)
    # Suppression de la dernière ligne (NaN)
    #df = df.dropna()

    return df


def load_data_opti(filepath, selected_days):
    # Paramètres de base
    rows_per_day = 96
    design_days = 2 * rows_per_day + 1

    dict_newindex = {
        'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)': 'Tout',
        'LIVING_UNIT1:Zone Air Temperature [C](TimeStep)': 'Tzone',
        'LIVING_UNIT1:Zone Air System Sensible Heating Rate [W](TimeStep)': 'Heating_living_unit',
        'LIVING_UNIT1:Zone Air System Sensible Cooling Rate [W](TimeStep)': 'Cooling_living_unit',
        'Fans:Electricity [J](TimeStep)': 'Pfans'
    }

    # Chargement en sautant les jours de dimensionnement
    df = pd.read_csv(filepath, sep=";", skiprows=range(1, design_days))
    df.columns = df.columns.str.strip()
    df.rename(columns=dict_newindex, inplace=True)

    # --- TRAITEMENT DU DATETIME (MM/DD HH:MM:SS -> 2025-MM-DD) ---
    # On ajoute l'année 2025 devant la chaîne
    df['Date/Time'] = df['Date/Time'].str.strip()
    # Gestion du 24:00:00 (EnergyPlus style) en le ramenant à 00:00:00
    df['Date/Time'] = df['Date/Time'].str.replace('24:00:00', '00:00:00')

    df['datetime'] = pd.to_datetime("2025/" + df['Date/Time'], format="%Y/%m/%d %H:%M:%S")

    # Si on a remplacé 24:00 par 00:00, ce point appartient techniquement au jour suivant
    # On ajuste les index pour que le 00:00:00 soit bien la fin du jour précédent ou le début du nouveau
    df.set_index('datetime', inplace=True)
    df = df.sort_index()

    # --- EXTRACTION POUR LES 12 JOURS ---
    data_12_days = {}

    for day in selected_days:
        start = pd.Timestamp(day + " 00:00:00")
        end = pd.Timestamp(day + " 23:45:00")

        try:
            day_slice = df.loc[start:end].copy()

            if len(day_slice) == 96:
                data_12_days[day] = {
                    'Tout': day_slice['Tout'].values,
                    'Tzone_init': day_slice['Tzone'].iloc[0],
                    'Pfans': day_slice['Pfans'].values/900 if 'Pfans' in day_slice else np.full(96, 100.0),
                    # 100W par défaut
                    'Tzone_real': day_slice['Tzone'].values  # Pour comparer après l'opti
                }
            else:
                print(f"⚠️ Jour {day} incomplet dans EnergyPlus ({len(day_slice)} points)")
        except KeyError:
            print(f"❌ Jour {day} absent du fichier EnergyPlus")

    return data_12_days


def compute_metrics(y_true, y_pred, train_time = None, test_time = None, dt_sec = 900):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

    errors = y_pred - y_true

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    max_error = np.max(np.abs(errors))
    mbe = np.mean(errors) # Montre si ton modèle : sousestime (MBE > 0) surestime (MBE < 0)

    r2 = r2_score(y_true, y_pred)
    var = np.var(errors)
    std = np.std(errors)

    cv_rmse = rmse / np.mean(y_true) * 100 #Seuils usuels : < 10 % → excellent < 15 % → acceptable


    metrics = {
        "MSE (°C²)": mse,
        "RMSE (°C)": rmse,
        "MAE (°C)": mae,
        "Max Error (°C)": max_error,
        "MBE / Bias (°C)": mbe,
        "R² (-)": r2,
        "CV(RMSE) (%)": cv_rmse,
        "Error Variance (°C²)": var,
        "Error Std Dev (°C)": std,
    }

    if train_time is not None:
        metrics["Train Time (s)"] = train_time
    if test_time is not None:
        metrics["Test Time (s)"] = test_time

    for k, v in metrics.items():
        print(f"{k:25s}: {v:.4f}")

    return metrics


def time_function(function, *args, **kwargs ):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def save_run_to_excel(
        filepath,
        model_name,
        metrics: dict,
        comment: str = ""
):

    filepath = Path(filepath)

    # Fusion paramètres + métriques
    row = {
        #"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        **metrics,
        "comment": comment
    }

    df_new = pd.DataFrame([row])


    if filepath.exists():
        df_old = pd.read_excel(filepath)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_excel(filepath, index=False)

def save_predictions(
    filepath,
    datetime_index,
    t_true,
    t_pred
):
    df = pd.DataFrame({
        "Time": datetime_index,
        "T_true": t_true,
        "T_pred": t_pred
    })

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_excel(filepath, index=False)



def create_run_folder(model_name, base_dir="results/"):
    run_name = f"{model_name}"
    run_path = os.path.join(base_dir, run_name)

    os.makedirs(run_path, exist_ok=True)
    return run_path


def agregate(runfile):

    runfile = runfile

    project_root = Path(__file__).resolve().parents[1]
    run_dir = project_root / "results" / runfile
    FILES = {
        "Benchmark": run_dir / "metrics_benchmark_rc.xlsx",
        "PySR": run_dir / "metrics_pysr.xlsx",
        "RL": run_dir / "metrics_rl.xlsx",
        "RC": run_dir / "metrics_rc.xlsx"
    }
    print("PROJECT_ROOT =", project_root)
    print("RUN_DIR =", run_dir)

    dfs = []

    for model, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f" Fichier introuvable : {path}")
        df = pd.read_excel(path)
        df["model"] = model
        dfs.append(df)

    final_df = pd.concat(dfs, ignore_index=True)

    output = Path(run_dir / "metrics_all_models.xlsx")
    final_df.to_excel(output, index=False)

    print(f" Fichier créé : {output}")

def tolatex(runfile):
    project_root = Path(__file__).resolve().parents[1]
    run_dir = project_root / "results" / runfile

    file_path = run_dir / "metrics_all_models.xlsx"
    df = pd.read_excel(file_path)

    #df = df.round(3)
    df_transposed = df.T
    # 1. Utiliser la première ligne comme en-tête
    df_transposed.columns = df_transposed.iloc[0]

    # 2. Supprimer la première ligne (qui est maintenant dans l'en-tête)
    # et réinitialiser l'index pour que "Models", "RC", etc. soit une colonne
    df_final = df_transposed.drop(df_transposed.index[0]).reset_index()
    # df_final = df_transposed.reset_index()

    # 3. Optionnel : renommer la colonne d'index si nécessaire
    df_final = df_final.rename(columns={'index': 'Modeles'})

    # 2. Convertir en LaTeX
    # index=False pour ne pas inclure la numérotation des lignes
    latex_code = df_final.to_latex(index=False, caption="Mon tableau", label="tab:mon_tableau", float_format = "%.3f")
    # ou float_format = "%.3f" dans to_latex
    # 3. Sauvegarder dans un fichier .tex
    with open(run_dir/"tableau.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)

    print("Conversion réussie ! Fichier 'tableau.tex' généré.")

