import numpy as np
import sympy as sp
import pandas as pd
import random
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
#a jour


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


def process_market_prices(filepath, seed=42):
    """
    Traite le fichier de prix ENTSO-E, sélectionne 12 jours et crée les vecteurs de prix.

    Args:
        filepath (str): Chemin vers le fichier CSV des prix.
        seed (int): Graine pour la reproductibilité du tirage aléatoire.

    Returns:
        tuple: (selected_days, dict_days_prices, df_full)
    """
    # 1. Chargement
    df = pd.read_csv(filepath)

    # 2. Nettoyage temporel
    df['datetime_str'] = df['MTU (CET/CEST)'].str.split(' - ').str[0]
    df['datetime_str'] = df['datetime_str'].str.replace(' (CET)', '', regex=False)
    df['datetime_str'] = df['datetime_str'].str.replace(' (CEST)', '', regex=False)

    df['datetime'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')
    df = df.drop_duplicates(subset=['datetime']).set_index('datetime').sort_index()

    # 3. Conversion EUR/MWh -> EUR/kWh
    df['price_eur_kwh'] = df['Day-ahead Price (EUR/MWh)'] / 1000

    # 4. Interpolation 15-min (gestion des rampes avant octobre)
    prices_series = df['price_eur_kwh'].copy()
    mask_paliers = prices_series.index < '2025-10-01'

    prices_to_interp = prices_series.copy()
    # On ne garde que les points "pile à l'heure" pour forcer la rampe
    prices_to_interp.loc[mask_paliers & (prices_to_interp.index.minute != 0)] = None

    df['prices_15min'] = prices_to_interp.interpolate(method='linear')

    # 5. Sélection aléatoire reproductible des 12 jours
    random.seed(seed)
    selected_days = [f"2025-{m:02d}-{random.randint(1, 28):02d}" for m in range(1, 13)]

    # 6. Extraction des vecteurs de 96 points
    dict_days_prices = {}
    for day in selected_days:
        start_ts = pd.Timestamp(day + " 00:00:00")
        end_ts = pd.Timestamp(day + " 23:45:00")
        try:
            data = df['prices_15min'].loc[start_ts:end_ts]
            if len(data) >= 96:
                dict_days_prices[day] = data.iloc[:96].values
                print(f"✅ Jour {day} : 96 points de prix extraits.")
        except KeyError:
            print(f"❌ Jour {day} : Données manquantes dans le fichier source.")

    return selected_days, dict_days_prices, df

def load_data_opti_new(filepath, selected_days):
    # Paramètres de base
    rows_per_day = 96
    design_days = 2 * rows_per_day + 1

    dict_newindex = { # LIVINGUNIT pour csv année classique  LIVING_UNIT1 dans le csv energyplus baseline (20-24)
        'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)': 'Tout',
        'LIVING_UNIT1:Zone Air Temperature [C](TimeStep)': 'Tzone',
        'LIVING_UNIT1:Zone Air System Sensible Heating Rate [W](TimeStep)': 'Heating_living_unit',
        'LIVING_UNIT1:Zone Air System Sensible Cooling Rate [W](TimeStep)': 'Cooling_living_unit',
        'Fans:Electricity [J](TimeStep)': 'Pfans',
        'LIVING_UNIT1:Zone Thermostat Heating Setpoint Temperature [C](TimeStep)': 'Tset_heat',
        'LIVING_UNIT1:Zone Thermostat Cooling Setpoint Temperature [C](TimeStep)': 'Tset_cool',
        'Heating:Electricity [J](TimeStep)': 'P_heating',
        'Cooling:Electricity [J](TimeStep)': 'P_cooling'
    }

    # Chargement en sautant les jours de dimensionnement
    df = pd.read_csv(filepath, sep=";", skiprows=range(1, design_days))
    df.columns = df.columns.str.strip()
    df.rename(columns=dict_newindex, inplace=True)

    # 1. Traitement spécifique du format EnergyPlus
    def fix_ep_datetime(row):
        date_part = row['Date/Time'].strip()
        if '24:00:00' in date_part:
            # On remplace 24h par 00h et on ajoutera un jour après conversion
            clean_date = date_part.replace('24:00:00', '00:00:00')
            return pd.to_datetime("2025/" + clean_date, format="%Y/%m/%d %H:%M:%S") + pd.Timedelta(days=1)
        else:
            return pd.to_datetime("2025/" + date_part, format="%Y/%m/%d %H:%M:%S")

    df['datetime'] = df.apply(fix_ep_datetime, axis=1)
    df.set_index('datetime', inplace=True)
    df = df.sort_index()

    data_12_days = {}
    for day in selected_days:
        # Pour le marché 00:00 -> 23:45, on prend :
        # - La ligne 00:00:00 (qui était le 24:00:00 de la veille) -> T_initial
        # - Les lignes 00:15:00 jusqu'à 23:45:00 -> Les 95 pas suivants
        start = pd.Timestamp(day + " 00:00:00")
        end = pd.Timestamp(day + " 23:45:00")

        day_slice = df.loc[start:end]
        if len(day_slice) >= 96:  # On a 00:00 + les 96 quarts d'heure
            # T_initial est la valeur à minuit pile (t=0)

            # Les vecteurs (Tout, Setpoints) pour l'optimisation (t=0 à 95)
            # On ignore le 00:00 pour les variables de flux car le premier
            # impact du HVAC se voit à 00:15
            data_12_days[day] = {
                'Tout': day_slice['Tout'].iloc[:96].values,
                'Tset_heat': day_slice['Tset_heat'].iloc[:96].values if 'Tset_heat' in day_slice.columns else np.ones(
                    96) * 20,
                'Tset_cool': day_slice['Tset_cool'].iloc[:96].values if 'Tset_cool' in day_slice.columns else np.ones(
                    96) * 24,
                #'Tset_heat': day_slice['Tset_heat'].iloc[:96].values,
                #'Tset_cool': day_slice['Tset_cool'].iloc[:96].values,
                'Tzone_init': day_slice['Tzone'].iloc[0],
                'Tzone_real': day_slice['Tzone'].values,
                'Pfans': day_slice['Pfans'].iloc[:96].values / 900
            }

    return data_12_days, df

def load_data_api(filepath):
    rows_per_day = 96
    design_days = 2 * rows_per_day + 1
    # Chargement en sautant les jours de dimensionnement
    df = pd.read_csv(filepath, sep=";", skiprows=range(1, design_days))
    df.columns = df.columns.str.strip()

    # 1. Traitement spécifique du format EnergyPlus
    def fix_ep_datetime(row):
        date_part = row['Date/Time'].strip()
        if '24:00:00' in date_part:
            # On remplace 24h par 00h et on ajoutera un jour après conversion
            clean_date = date_part.replace('24:00:00', '00:00:00')
            return pd.to_datetime("2017/" + clean_date, format="%Y/%m/%d %H:%M:%S") + pd.Timedelta(days=1)
        else:
            return pd.to_datetime("2017/" + date_part, format="%Y/%m/%d %H:%M:%S")

    df['datetime'] = df.apply(fix_ep_datetime, axis=1)
    df.set_index('datetime', inplace=True)
    data_annual = df.sort_index()
    return data_annual


def calculate_average_efficiencies(df_annual):

    # éviter les divisions par zéro
    mask_heating = (df_annual['P_heating'] > 0)
    mask_cooling = (df_annual['P_cooling'] > 0)

    # 2. Calcul du rendement instantané : Q_thermique / P_electrique
    eta_h_series = df_annual.loc[mask_heating, 'Heating_living_unit'] / (df_annual.loc[mask_heating, 'P_heating']/900)
    eta_c_series = df_annual.loc[mask_cooling, 'Cooling_living_unit'] / (df_annual.loc[mask_cooling, 'P_cooling']/900)

    avg_eta_h = eta_h_series.mean()
    avg_eta_c = eta_c_series.mean()

    print(f"--- Analyse des rendements annuels (EnergyPlus) ---")
    print(f"Moyenne COP (Chauffage) : {avg_eta_h:.3f}")
    print(f"Moyenne EER (Refroidissement) : {avg_eta_c:.3f}")

    return avg_eta_h, avg_eta_c

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


def agregate(runfile, step = "1step"):

    runfile = runfile

    project_root = Path(__file__).resolve().parents[1]
    run_dir = project_root / "outputs"

    #run_dir = project_root / "results" / runfile
    FILES = {
    #    #"Benchmark": run_dir / "metrics_benchmark_rc.xlsx",
    #    "PySR": run_dir / f"metrics_PySR_{step}.xlsx",
    #    "RL": run_dir / f"metrics_Linear_{step}.xlsx",
    #    "RC": run_dir / f"metrics_RC_{step}.xlsx"
        "PySR_exp": run_dir / "pysr_exp_n150_p20" /f"metrics_{step}.xlsx",
        "PySR_cube": run_dir / "pysr_quad_iter200_pop30" / f"metrics_{step}.xlsx",

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

    output = Path(run_dir / f"metrics_all_models_{step}.xlsx")
    final_df.to_excel(output, index=False)

    print(f" Fichier créé : {output}")

def tolatex(runfile, file = "results", step = "1step"):
    project_root = Path(__file__).resolve().parents[1]
    #run_dir = project_root / file / runfile   # pour les data in results
    run_dir = project_root / file

    file_path = run_dir / f"metrics_all_models_{step}.xlsx"
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
    with open(run_dir/f"tableau_{step}.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)

    print("Conversion réussie ! Fichier 'tableau.tex' généré.")

def to_latex(version, name, runfile, file_path):

    df = pd.read_excel(f"api{version}/{name}_{runfile}/{file_path}.xlsx", sheet_name=0)
    if file_path == "Validation_EPlus_VS_Opti":
        colonnes_cibles = ['Day', 'Cost_Opti', 'Cost_Real', 'Difference_Euro', 'RMSE_Temp', 'RMSE_Power_W',
                       'MAPE_Power_Pct', 'Energy_Opti_kWh', 'Energy_Real_kWh']
    else:
        colonnes_cibles = ['Day', 'Cost_Real', 'RMSE_Temp', 'RMSE_Power_W',
                           'MAPE_Power_Pct', 'Energy_Real_kWh']
    df = df[colonnes_cibles]
    df.columns = [
        str(c).replace('_', ' ')
        .replace('Temperature', 'Temp.')
        .replace('Power Pct', 'P (%)')
        .replace('Difference', 'Diff.')
        .replace('Euro', '€')
        .replace('Energy', 'Energ')
        .replace('Real', 'ex-post')
        for c in df.columns
    ]
    num_cols = len(df.columns)

    # On crée un format : la première colonne (Day) est alignée à gauche (l),
    # toutes les autres sont des colonnes fines centrées à largeur automatique.
    column_format = "l" + "c" * (num_cols - 1)


    # 2. Convertir en LaTeX
    # index=False pour ne pas inclure la numérotation des lignes
    latex_code = df.to_latex(
        index=False,
        caption=f"Métriques globales - Modèle {name} ({runfile})",
        label=f"tab:{name}_{runfile}_{file_path}",
        float_format = "%.2f",
        column_format=column_format  # On force l'alignement serré
    )
    latex_code = latex_code.replace(
        "\\begin{tabular}",
        "\\begin{small}\n\\begin{tabular}"
    )
    latex_code = latex_code.replace(
        "\\end{tabular}",
        "\\end{tabular}\n\\end{small}"
    )
    # ou float_format = "%.3f" dans to_latex
    # 3. Sauvegarder dans un fichier .tex
    with open(f"api{version}/{name}_{runfile}/{file_path}.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)

    print("Conversion réussie ! Fichier 'tableau.tex' généré.")

def export_opti_results_to_excel(df_summary, results_all_days, output_dir ="opti", output_path = "file.xlsx"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    with pd.ExcelWriter(Path(output_dir)/output_path, engine='xlsxwriter') as writer:
        # Feuille 1 : Résumé global
        df_summary.to_excel(writer, sheet_name='Resume_Couts', index=False)

        # 12 Feuilles : Une par jour
        for day, values in results_all_days.items():
            # Préparation du DataFrame spécifique au jour (alignement 97 points)
            # On ajoute NaN à la fin pour les données de flux (96 -> 97)
            data_day = {
                'Timestep': range(97),
                'T_zone_[°C]': values['T_zone'],
                'P_heating_[W]': values['P_heating'] + [np.nan],
                'P_cooling_[W]': values['P_cooling'] + [np.nan],
                'Prix_[EUR/kWh]': list(values['Prices']) + [np.nan],
                'Cout_Instant_[EUR]': values['Step_Costs'] + [np.nan],
                'T_borne_low': values['T_borne_low'] + [np.nan],
                'T_borne_high': values['T_borne_high'] + [np.nan]
            }

            df_day = pd.DataFrame(data_day)

            # On utilise la date comme nom de feuille (max 31 caractères pour Excel)
            sheet_name = f"Day_{day}"
            df_day.to_excel(writer, sheet_name=sheet_name, index=False)


def save_plot_day(day, results_all_days, output_dir="opti/results_image"):
    """
    Génère et sauvegarde les 3 graphiques pour un jour spécifique.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Paramètres communs
    res = results_all_days[day]
    hours_p_opt_post = np.arange(0, 24, 0.25)
    hours_t_opt = np.arange(0, 24.25, 0.25)
    xticks = np.arange(0, 25, 2)
    Ph_max, Pc_max = res['P_h_max'], res['P_c_max']
    tmin, tmax = res['T_borne_low'], res['T_borne_high']

    # --- 1. FIGURE TEMPÉRATURES ---
    plt.figure()
    plt.plot(hours_t_opt, res['T_zone'], color='blue', label=f"T_zone {day}", linewidth=2.5)
    plt.plot(hours_p_opt_post, tmin, color='green', linestyle='--', alpha=0.5, label="T_min")
    plt.plot(hours_p_opt_post, tmax, color='red', linestyle='--', alpha=0.5, label="T_max")
    plt.title(f"Courbe de température zone - {day}")
    plt.ylabel("Température [°C]")
    plt.xlabel("Heure [h]")
    plt.xticks(xticks)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Temp_Zone_{day}.png")
    plt.close()  # Ferme la figure pour libérer la RAM

    # --- 2. FIGURE PUISSANCES HVAC ---

    p_heat_opt_97 = np.append(res['P_heating'], res['P_heating'][-1])
    p_cool_opt_97 = np.append(-np.array(res['P_cooling']), -res['P_cooling'][-1])

    plt.figure() #figsize=(10, 5)
    p_net = np.array(p_heat_opt_97 - p_cool_opt_97)
    plt.step(hours_t_opt, p_net, where='post', color='orange', label="P_net (heating > 0 et cooling < 0)", linewidth=2)
    plt.axhline(y=Ph_max, color='red', linestyle='--', alpha=0.3, label="P_max Heat")
    plt.axhline(y=-Pc_max, color='blue', linestyle='--', alpha=0.3, label="P_max Cool")
    plt.title(f"Profil Puissance HVAC - {day}")
    plt.ylabel("Puissance [W]")
    plt.xlabel("Heure [h]")
    plt.xticks(xticks)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Puissance_HVAC_{day}.png")
    plt.close()

    # --- 3. FIGURE PRIX ET COÛTS ---
    plt.figure()
    p_tot = np.array(res['P_heating']) + np.array(res['P_cooling'])
    step_costs = (res['Prices'] * p_tot / 1000 * 0.25)
    step_costs = np.append(step_costs, step_costs[-1])
    res['Prices'] = np.append(res['Prices'], res['Prices'][-1])
    plt.step(hours_t_opt, res['Prices'], where='post', color='blue', linewidth=1.5, label="Prix marché [€/kWh]")
    plt.step(hours_t_opt, step_costs, where='post', color='black', alpha=0.7, linewidth=2, label="Coût [€/15min]")
    plt.fill_between(hours_t_opt, step_costs, step='post', color='black', alpha=0.1)

    plt.title(f"Analyse Économique - {day}")
    plt.ylabel("Valeur [€]")
    plt.xlabel("Heure [h]")
    plt.xticks(xticks)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Prix_Couts_{day}.png")
    plt.close()

    plt.close('all')


def plot_comparison_results_api(day_str, res_opt, df_cleaned, df_baseline, output_dir=None):
    """
    Génère les graphiques comparatifs en utilisant le DataFrame déjà filtré.
    attention energyplus donne les températures instantanné au timestep mais les puissances sont pour l'intervalle précédent.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- préparation des axes temporels
    hours_97 = np.arange(0, 24.25, 0.25)

    # Pour l'optimisation
    t_opt_97 = res_opt['T_zone'][:97]
    # Pour EnergyPlus Réel et Baseline
    t_real_97 = df_cleaned['Zone Air Temperature,LIVING_UNIT1'].iloc[:97].values
    t_base_97 = df_baseline['LIVING_UNIT1:Zone Air Temperature [C](TimeStep)'].iloc[:97].values

    # Extension des bornes de confort floues (Slacks) de 96 à 97 points pour le style 'post'
    low_borne_97 = np.append(res_opt['T_borne_low'], res_opt['T_borne_low'][-1])
    high_borne_97 = np.append(res_opt['T_borne_high'], res_opt['T_borne_high'][-1])

    # --- PUISSANCES ET FLUX (Données d'intervalles : 96 pas rééchantillonnés par le DEBUT de l'intervalle) ---
    # On force le rééchantillonnage avec label='left' pour que la consommation de l'intervalle (ex: lue à 00:15)
    # soit correctement positionnée à son heure de début (00:00) sur l'axe X.
    df_resampled_real = df_cleaned.resample('15min', label='left').mean().iloc[:96]
    df_resampled_base = df_baseline.resample('15min', label='left').mean().iloc[:96]

    # Figure 2 : Flux thermiques (Q_hvac) - 96 points étendus à 97 pour le tracé 'post'
    q_opt_96 = (1.86 * np.array(res_opt['P_heating'])) - (3.73 * np.array(res_opt['P_cooling']))
    q_opt_97 = np.append(q_opt_96, q_opt_96[-1])

    q_real_96 = (df_resampled_real['Zone Air System Sensible Heating Rate,LIVING_UNIT1'] -
                 df_resampled_real['Zone Air System Sensible Cooling Rate,LIVING_UNIT1']).values
    q_real_97 = np.append(q_real_96, q_real_96[-1])

    q_base_96 = (df_resampled_base['LIVING_UNIT1:Zone Air System Sensible Heating Rate [W](TimeStep)'] -
                 df_resampled_base['LIVING_UNIT1:Zone Air System Sensible Cooling Rate [W](TimeStep)']).values
    q_base_97 = np.append(q_base_96, q_base_96[-1])

    # Figure 3 : Puissances électriques - 96 points étendus à 97 pour le tracé 'post'
    p_heat_opt_97 = np.append(res_opt['P_heating'], res_opt['P_heating'][-1])
    p_cool_opt_97 = np.append(-np.array(res_opt['P_cooling']), -res_opt['P_cooling'][-1])

    dt_sec = 900
    p_heat_ep_96 = (df_resampled_real['Heating:Electricity'] / dt_sec).values
    p_cool_ep_96 = -(df_resampled_real['Cooling:Electricity'] / dt_sec).values
    p_heat_ep_97 = np.append(p_heat_ep_96, p_heat_ep_96[-1])
    p_cool_ep_97 = np.append(p_cool_ep_96, p_cool_ep_96[-1])

    p_heat_base_96 = (df_resampled_base['Heating:Electricity [J](TimeStep)'] / dt_sec).values
    p_cool_base_96 = -(df_resampled_base['Cooling:Electricity [J](TimeStep)'] / dt_sec).values
    p_heat_base_97 = np.append(p_heat_base_96, p_heat_base_96[-1])
    p_cool_base_97 = np.append(p_cool_base_96, p_cool_base_96[-1])

    # Figure 4 : Coûts au quart d'heure - 96 points étendus à 97 pour le tracé 'post'
    cost_opt_97 = np.append(res_opt['Step_Costs'], res_opt['Step_Costs'][-1])
    cost_real_97 = np.append(df_resampled_real['step_cost_real'].values, df_resampled_real['step_cost_real'].values[-1])
    cost_base_97 = np.append(df_resampled_base['step_cost_real'].values, df_resampled_base['step_cost_real'].values[-1])


    #--- création graphes ---
    # --- FIGURE 1 : COMPARAISON TEMPÉRATURES ET BORNES DYNAMIQUES (SLACKS) ---
    plt.figure()
    plt.plot(hours_97, t_opt_97, 'r--', label="T_zone Opti (Consigne)", linewidth=2)
    plt.plot(hours_97, t_real_97, 'b-', label="T_zone EnergyPlus (Réel Ex-post)", alpha=0.8)
    plt.plot(hours_97, t_base_97, 'k-', label="Baseline (Thermostat Standard)", alpha=0.4)

    # Bornes en escalier (slacks incluses) qui s'étirent proprement jusqu'à 24h00
    plt.step(hours_97, low_borne_97, where='post', color='darkred', linestyle=':', alpha=0.6,
             label="Bornes Confort Dynamiques")
    plt.step(hours_97, high_borne_97, where='post', color='darkred', linestyle=':', alpha=0.6)

    plt.title(f"Profils des Températures de Zone - {day_str}", fontweight='bold')
    plt.ylabel("Température [°C]")
    plt.xlabel("Heure [h]")
    plt.xlim(0, 24)
    plt.xticks(np.arange(0, 25, 2))
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Comp_Temp_{day_str}.png", dpi=300)
    plt.savefig(f"{output_dir}/Comp_Temp_{day_str}.pdf")
    plt.close()

    # --- FIGURE 2 : COMPARAISON PUISSANCES THERMIQUES (Q_hvac) ---
    plt.figure()
    plt.step(hours_97, q_opt_97, where='post', label="Q_hvac Opti (Théorique)", color='red', alpha=0.6, linewidth=1.5)
    plt.step(hours_97, q_real_97, where='post', label="Q_hvac EnergyPlus (Réel Ex-post)", color='blue', alpha=0.7,
             linewidth=1.2)
    plt.step(hours_97, q_base_97, where='post', label="Q_hvac Baseline", color='gray', alpha=0.4)

    plt.title(f"Profils des Puissances Thermiques - {day_str}", fontweight='bold')
    plt.ylabel("Puissance Thermique [W]")
    plt.xlabel("Heure [h]")
    plt.xlim(0, 24)
    plt.xticks(np.arange(0, 25, 2))
    plt.axhline(0, color='black', linewidth=0.8, alpha=0.3)
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Comp_Power_{day_str}.png", dpi=300)
    plt.savefig(f"{output_dir}/Comp_Power_{day_str}.pdf")
    plt.close()

    # --- FIGURE 3 : PUISSANCES ÉLECTRIQUES APPELÉES ---
    plt.figure()
    # Tracés Optimisation (Paliers parfaits 'post')
    plt.step(hours_97, p_heat_opt_97, where='post', label="P_heat Opti", color='red', alpha=0.5)
    plt.step(hours_97, p_cool_opt_97, 'r--', where='post', label="P_cool Opti", alpha=0.5)

    # Tracés EnergyPlus Réel Ex-post (Paliers lissés corrigés à 900s)
    plt.step(hours_97, p_heat_ep_97, where='post', label="P_heat E+ (Réel)", color='blue', alpha=0.8, linewidth=1.2)
    plt.step(hours_97, p_cool_ep_97, 'b--', where='post', label="P_cool E+ (Réel)", alpha=0.8, linewidth=1.2)

    # Tracés Baseline
    plt.step(hours_97, p_heat_base_97, where='post', label="P_heat Baseline", color='black', alpha=0.3)
    plt.step(hours_97, p_cool_base_97, where='post', label="P_cool Baseline", color='grey', alpha=0.3)

    plt.title(f"Profils des Puissances Électriques HVAC - {day_str}", fontweight='bold')
    plt.ylabel("Puissance Électrique [W] (cooling < 0)")
    plt.xlabel("Heure de la journée [h]")
    plt.xlim(0, 24)
    plt.xticks(np.arange(0, 25, 2))
    plt.axhline(0, color='black', linewidth=0.8, alpha=0.3)
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Comp_Electric_Power_{day_str}.png", dpi=300)
    plt.savefig(f"{output_dir}/Comp_Electric_Power_{day_str}.pdf")
    plt.close()

    # --- FIGURE 4 : COÛTS FINANCIERS AU QUART D'HEURE ---
    plt.figure()
    plt.step(hours_97, cost_real_97, where='post', color='blue', linewidth=1.5, label="Coût Ex-post E+")
    plt.step(hours_97, cost_base_97, where='post', color='grey', linewidth=1.5, label="Coût Baseline E+")
    plt.step(hours_97, cost_opt_97, where='post', color='red', alpha=0.7, linewidth=1.5, label="Coût Prédit Opti")

    plt.title(f" Coûts d'Électricité par Quart d'Heure - {day_str}", fontweight='bold')
    plt.ylabel("Coût du pas de temps [€]")
    plt.xlabel("Heure de la journée [h]")
    plt.xlim(0, 24)
    plt.xticks(np.arange(0, 25, 2))
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Couts_{day_str}.png", dpi=300)
    plt.savefig(f"{output_dir}/Couts_{day_str}.pdf")

    # Nettoyage absolu de la mémoire Matplotlib pour le jour ou modèle suivant
    plt.close('all')



def plot_global_costs_bar_chart(all_stats_list, all_stats_list_sans_opti, name, runfile, output_dir="apiV4"):
    """
    Génère un graphique en bâtonnets comparant le coût total journalier
    entre la Baseline, l'Optimisation et l'Ex-post EnergyPlus pour les 12 jours.
    """
    # 1. Extraction et alignement des données
    days = [stat['Day'] for stat in all_stats_list]
    cost_opti = [stat['Cost_Opti'] for stat in all_stats_list]
    cost_expost = [stat['Cost_Real'] for stat in all_stats_list]

    # Mapper les coûts de la baseline par jour pour garantir la correspondance
    baseline_dict = {stat['Day']: stat['Cost_Real'] for stat in all_stats_list_sans_opti}
    cost_baseline = [baseline_dict.get(day, 0.0) for day in days]

    # 2. Configuration géométrique des barres
    x = np.arange(len(days))  # Emplacement des étiquettes des jours
    width = 0.25  # Largeur de chaque bâtonnet

    fig, ax = plt.subplots(figsize=(14, 6))

    # Tracé des 3 groupes de bâtonnets
    rects1 = ax.bar(x - width, cost_baseline, width, label='Baseline EnergyPlus', color='gray', alpha=0.7)
    rects2 = ax.bar(x, cost_opti, width, label='Optimisation (Théorique)', color='red')
    rects3 = ax.bar(x + width, cost_expost, width, label='Ex-post EnergyPlus (Réel)', color='blue')

    # 3. Personnalisation des axes et légendes
    ax.set_ylabel('Coût Total du Jour [€]', fontsize=11, fontweight='bold')
    ax.set_title(f'Comparaison des Coûts Totaux Journaliers - Modèle {name}', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45, ha='right')
    ax.grid(True, linestyle=':', alpha=0.6, axis='y')
    ax.legend(loc='upper right', fontsize=10)

    # 4. Ajout des valeurs au-dessus des barres pour la lisibilité
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}€',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Décalage vertical de 3 points
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Comparaison_Globale_Couts_12jours.pdf")
    plt.savefig(f"{output_dir}/Comparaison_Globale_Couts_12jours.png")
    plt.close('all')

def export_validation_to_excel(all_stats, all_df_days, output_path="opti/Validation_EPlus_VS_Opti.xlsx"):
    """
    all_stats: liste des dictionnaires de stats
    all_df_days: dictionnaire {day_str: df_day}
    """
    # 1. Création du résumé global
    df_summary = pd.DataFrame(all_stats)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # Feuille 1 : Résumé
        df_summary.to_excel(writer, sheet_name='Resume_Global', index=False)

        # Feuilles suivantes : Uniquement les données du jour (sans warmup)
        for day_str, df_day in all_df_days.items():
            # On peut supprimer la colonne 'datetime' si on veut un Excel plus léger,
            # mais c'est souvent utile de la garder.
            sheet_name = f"Details_{day_str}"
            df_day.to_excel(writer, sheet_name=sheet_name, index=True)

    print(f"✅ Rapport Excel généré (sans warmup) : {output_path}")


def analyze_variable_timestep_results_sans_opti(day_str, results_all_days, csv_dir="opti/Eplus_run_20_24"):
    """
    Analyse les résultats E+ à pas de temps variables et les compare à l'opti.
    """
    csv_path = os.path.join(csv_dir, f"model_annee_classique_20_24.csv")
    df = pd.read_csv(csv_path, sep=';')

    # --- ÉTAPE DE CORRECTION DU TIMESTEP ---
    # On remplace "02/29" par "03/01" uniquement dans la colonne Timestep
    # pour que Pandas accepte de le lire en année 2017
    # 1. Conversion du temps en datetime (en forçant l'année 2017 pour matcher l'API)
    # colonne timestep (format "MM/DD HH:MM")
    # 1. Gérer le "24:00:00" d'EnergyPlus
    # On identifie les lignes à minuit
    mask_midnight = df['Date/Time'].str.contains("24:00:00")
    df['Timestep_clean'] = df['Date/Time'].str.replace("24:00:00", "00:00:00")
    df['datetime'] = pd.to_datetime("2017/" + df['Timestep_clean'].str.strip(), format="%Y/%m/%d %H:%M:%S")
    # Pour les lignes qui étaient à 24:00:00, on ajoute 1 jour car c'est le minuit du lendemain
    df.loc[mask_midnight, 'datetime'] = df.loc[mask_midnight, 'datetime'] + pd.Timedelta(days=1)

    # 2. Filtrer pour ne garder que le jour d'intérêt (exclure le warmup)
    target_date = pd.to_datetime(day_str).replace(year=2017)
    next_date = target_date + pd.Timedelta(days=1)
    # df_day = df[df['datetime'].dt.date == target_date.date()].copy()
    df_day = df[(df['datetime'] >= target_date) & (df['datetime'] <= next_date)].copy()

    # 3. Calcul du dt réel en heures pour chaque ligne
    # On calcule la différence de temps avec la ligne suivante
    df_day['dt_hours'] = df_day['datetime'].diff().dt.total_seconds().shift(-1) / 3600
    # La dernière ligne n'a pas de suivante, on peut mettre 0.25 (15 min) par défaut
    df_day['dt_hours'].fillna(0.25, inplace=True)
    df_day['dt_seconds'] = df_day['dt_hours'] * 3600

    energy_cols = [
        'Heating:Electricity [J](TimeStep)',
        'Cooling:Electricity [J](TimeStep)',
        'Fans:Electricity [J](TimeStep) '
    ]
    df_day.loc[df_day['datetime'] == target_date, energy_cols] = 0.0
    # 4. Calcul du coût réel avec prix dynamique
    # Il faut mapper le prix de l'optimisation (qui est fixe par 15min)
    # sur chaque micro-pas de temps de EnergyPlus
    res_opt = results_all_days[day_str]
    prices_96 = res_opt['Prices']  # Liste de 96 prix

    def get_price_for_time(dt):
        if dt == next_date:
            return prices_96[-1]
        idx = int((dt.hour * 60 + dt.minute) // 15)
        return prices_96[min(idx, 95)]

    df_day['current_price'] = df_day['datetime'].apply(get_price_for_time)

    df_day = df_day.set_index('datetime')
    df_numeric = df_day.select_dtypes(include=[np.number])

    # Conversion Énergie [J] -> Puissance Moyenne [W] fixe sur le bloc de 15 min
    df_numeric['P_total_W'] = (df_numeric['Heating:Electricity [J](TimeStep)'] +
                               df_numeric['Cooling:Electricity [J](TimeStep)'] +
                               df_numeric['Fans:Electricity [J](TimeStep) ']) / 900.0

    # Calcul du coût exact par sous-pas (la ligne de minuit ajoutera 0€)
    df_numeric['step_cost_real'] = (df_numeric['P_total_W'] / 1000.0) * df_numeric['dt_hours'] * df_numeric['current_price']
    total_cost_day_ep = df_numeric['step_cost_real'].sum()

    # Intégration temporelle pour l'énergie totale cumulée [Joules] du jour J
    total_energy_j = (df_numeric['P_total_W'] * df_numeric['dt_seconds']).sum()
    total_energy_kwh = total_energy_j / 3600000.0
    avg_power_w = total_energy_j / (24 * 3600)

    df_resampled = df_numeric.resample('15min').mean().iloc[:97]
    # Rééchantillonnage pour la température (RMSE propre)
    # On prend la moyenne de température sur chaque 15 min pour comparer à l'opti

    p_real_96 = df_resampled['P_total_W'].iloc[1:97].values
    t_real_97 = df_resampled['LIVING_UNIT1:Zone Air Temperature [C](TimeStep)'].values
    t_opt_97 = np.array(res_opt['T_zone'][:97])

    # Métriques de la Baseline par rapport à la trajectoire de l'optimisation (Utile pour l'analyse)
    p_tot_opt = np.array(res_opt['P_heating']) + np.array(res_opt['P_cooling'])
    rmse_power = np.sqrt(np.mean((p_tot_opt - p_real_96) ** 2))
    mape_power = np.mean(np.abs((p_real_96 - p_tot_opt) / (p_real_96 + 1e-5))) * 100
    rmse_temp = np.sqrt(np.mean((t_opt_97 - t_real_97) ** 2))
    stats = {
        'Day': day_str,
        'Cost_Real': total_cost_day_ep,
        'RMSE_Temp': rmse_temp,
        'RMSE_Power_W': rmse_power,
        'MAPE_Power_Pct': mape_power,
        'Energy_Real_kWh': total_energy_kwh,
        'P_Avg_Real_W': avg_power_w
    }
    return stats, df_resampled

def analyze_variable_timestep_results(day_str, results_all_days, name, csv_dir=None):
    """
    Analyse les résultats E+ à pas de temps variables et les compare à l'opti.
    """
    csv_path = os.path.join(csv_dir, f"validation_EP_{name}_{day_str}.csv")
    df = pd.read_csv(csv_path, sep=';')

    # --- ÉTAPE DE CORRECTION DU TIMESTEP ---
    df['Timestep'] = df['Timestep'].str.replace("02/29", "03/01")
    df['datetime'] = pd.to_datetime("2017/" + df['Timestep'], format="%Y/%m/%d %H:%M")

    # Définition des frontières temporelles de la journée cible
    target_date = pd.to_datetime(day_str).replace(year=2017)
    next_date = target_date + pd.Timedelta(days=1)

    # --- FILTRAGE DES INTERVALLES COMPRENANT MINUIT INITIAL ---
    # On utilise '>=' pour conserver la ligne de minuit pile (00:00:00) du jour cible
    df_day = df[(df['datetime'] >= target_date) & (df['datetime'] <= next_date)].copy()

    # Calcul du dt réel en heures pour chaque ligne
    df_day['dt_hours'] = df_day['datetime'].diff().dt.total_seconds().shift(-1) / 3600
    df_day['dt_hours'].fillna(0.25, inplace=True)
    df_day['dt_seconds'] = df_day['dt_hours'] * 3600

    # --- NEUTRALISATION DE L'ÉNERGIE DE LA VEILLE (Ligne 00:00:00) ---
    # On force à 0 les compteurs de la toute première ligne à minuit pile,
    # car cette énergie appartient au jour précédent.
    energy_cols = ['Heating:Electricity', 'Cooling:Electricity', 'Fans:Electricity']
    df_day.loc[df_day['datetime'] == target_date, energy_cols] = 0.0

    res_opt = results_all_days[day_str]
    prices_96 = res_opt['Prices']

    def get_price_for_time(dt):
        if dt == next_date:
            return prices_96[-1]
        idx = int((dt.hour * 60 + dt.minute) // 15)
        return prices_96[min(idx, 95)]

    df_day['current_price'] = df_day['datetime'].apply(get_price_for_time)

    # --- SÉLECTION ET TRAITEMENT NUMÉRIQUE ---
    df_day = df_day.set_index('datetime')
    df_numeric = df_day.select_dtypes(include=[np.number])

    # Conversion Énergie [J] -> Puissance Moyenne [W] du bloc de 15 minutes
    df_numeric['P_total_W'] = (df_numeric['Heating:Electricity'] +
                               df_numeric['Cooling:Electricity'] +
                               df_numeric['Fans:Electricity']) / 900.0

    # Calcul du coût exact intégré par sous-pas de temps (la ligne 00:00:00 ajoutera 0€)
    df_numeric['step_cost_real'] = (df_numeric['P_total_W'] / 1000.0) * df_numeric['dt_hours'] * df_numeric[
        'current_price']
    total_cost_day_ep = df_numeric['step_cost_real'].sum()

    # Intégration temporelle pour l'énergie totale cumulée [Joules] du jour J
    total_energy_j = (df_numeric['P_total_W'] * df_numeric['dt_seconds']).sum()
    total_energy_kwh = total_energy_j / 3600000.0
    avg_power_w = total_energy_j / (24 * 3600)

    # --- CALCUL OPTIMISATION POUR COMPARAISON ---
    p_tot_opt = np.array(res_opt['P_heating']) + np.array(res_opt['P_cooling'])
    energy_opt_kwh = (p_tot_opt * 0.25).sum() / 1000.0
    avg_power_opt_w = p_tot_opt.mean()

    # --- RÉÉCHANTILLONNAGE SYNCHRONISÉ ---
    # label='left' associe la fin de l'intervalle au début (ex: 00:15 devient indexé à 00:00)
    df_resampled = df_numeric.resample('15min').mean().iloc[:97]

    # Extraction des vecteurs de comparaison (96 points pour les puissances)
    p_real_96 = df_resampled['P_total_W'].iloc[1:97].values
    t_real_97 = df_resampled['Zone Air Temperature,LIVING_UNIT1'].values
    t_opt_97 = np.array(res_opt['T_zone'][:97])

    # --- CALCUL DES MÉTRIQUES STATISTIQUES ---
    rmse_power = np.sqrt(np.mean((p_tot_opt - p_real_96) ** 2))
    mape_power = np.mean(np.abs((p_real_96 - p_tot_opt) / (p_real_96 + 1e-5))) * 100
    rmse_temp = np.sqrt(np.mean((t_opt_97 - t_real_97) ** 2))

    stats = {
        'Day': day_str,
        'Cost_Opti': res_opt['total_Cost'],
        'Cost_Real': total_cost_day_ep,
        'Difference_Euro': total_cost_day_ep - res_opt['total_Cost'],
        'RMSE_Temp': rmse_temp,
        'RMSE_Power_W': rmse_power,
        'MAPE_Power_Pct': mape_power,
        'Energy_Opti_kWh': energy_opt_kwh,
        'Energy_Real_kWh': total_energy_kwh,
        'P_Avg_Opti_W': avg_power_opt_w,
        'P_Avg_Real_W': avg_power_w
    }

    return stats, df_resampled


def plot_model_cuts_comparison(model_linear_func, model_quadratic_func, output_dir="apiV2/comparaisons_modeles"):
    """
    Génère des coupes physiques en 2D pour comparer visuellement le comportement
    d'un modèle linéaire et d'un modèle quadratique fournis en arguments.

    Parameters:
    -----------
    model_linear_func : function
        Fonction Python acceptant (T, Tout, Q) et renvoyant T_next pour le modèle linéaire.
    model_quadratic_func : function
        Fonction Python acceptant (T, Tout, Q) et renvoyant T_next pour le modèle quadratique.
    output_dir : str
        Dossier de sauvegarde des graphiques.
    """
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # COUPE 1 : SENSIBILITÉ À LA PUISSANCE HVAC (Q) EN HIVER
    # On fixe T_zone = 20°C, T_out = -5°C (Grand froid) et on fait varier Q
    # =========================================================================
    Q_range = np.linspace(-7120, 7120, 500)  # Plage complète de la PAC [W]
    T_fixed = 20.0
    Tout_fixed = -5.0

    T_next_lin_Q = [model_linear_func(T_fixed, Tout_fixed, q) for q in Q_range]
    T_next_quad_Q = [model_quadratic_func(T_fixed, Tout_fixed, q) for q in Q_range]

    plt.figure(figsize=(9, 5))
    plt.plot(Q_range, T_next_lin_Q, label="Modèle Linéaire", color="red", linestyle="--", linewidth=1.5)
    plt.plot(Q_range, T_next_quad_Q, label="Modèle Quadratique", color="blue", linewidth=2)
    plt.title(f"Coupe Thermique : Impact de Qhvac ($Q$) à $T_{{in}}$={T_fixed}°C et $T_{{out}}$={Tout_fixed}°C")
    plt.xlabel("Puissance Thermique HVAC $Q$ [W] (Chauffage > 0, Clim < 0)")
    plt.ylabel("$T_{{next}}$ après 15 min [°C]")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Coupe_Sensibilite_Q_Hiver.pdf", dpi=300)
    plt.savefig(f"{output_dir}/Coupe_Sensibilite_Q_Hiver.png", dpi=300)
    plt.close()

    # =========================================================================
    # COUPE 2 : SENSIBILITÉ À LA TEMPÉRATURE EXTÉRIEURE (Tout)
    # On fixe T_zone = 21°C, Q = 0W (Bâtiment passif) et on fait varier Tout
    # =========================================================================
    Tout_range = np.linspace(-10, 35, 500)  # De l'hiver à l'été [°C]
    T_fixed_2 = 21.0
    Q_fixed_2 = 0.0

    T_next_lin_Tout = [model_linear_func(T_fixed_2, tout, Q_fixed_2) for tout in Tout_range]
    T_next_quad_Tout = [model_quadratic_func(T_fixed_2, tout, Q_fixed_2) for tout in Tout_range]

    plt.figure(figsize=(9, 5))
    plt.plot(Tout_range, T_next_lin_Tout, label="Modèle Linéaire (Pente fixe)", color="red", linestyle="--",
             linewidth=1.5)
    plt.plot(Tout_range, T_next_quad_Tout, label="Modèle Quadratique (Courbure)", color="blue", linewidth=2)
    plt.title(f"Coupe Thermique : Dérive Passive ($Q=0$W) à $T_{{in}}$={T_fixed_2}°C")
    plt.xlabel("Température Extérieure $T_{out}$ [°C]")
    plt.ylabel("$T_{{next}}$ après 15 min [°C]")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Coupe_Sensibilite_Tout_Passif.pdf", dpi=300)
    plt.savefig(f"{output_dir}/Coupe_Sensibilite_Tout_Passif.png", dpi=300)
    plt.close()

    # =========================================================================
    # COUPE 3 : APPORTS DU TERME CROISÉ (T * Tout)
    # On fixe Q = 3000W (Chauffage moyen), Tout = 5°C et on fait varier T_zone
    # =========================================================================
    T_range = np.linspace(15, 26, 500)  # Plage de confort élargie [°C]
    Tout_fixed_3 = 5.0
    Q_fixed_3 = 3000.0

    T_next_lin_T = [model_linear_func(t, Tout_fixed_3, Q_fixed_3) for t in T_range]
    T_next_quad_T = [model_quadratic_func(t, Tout_fixed_3, Q_fixed_3) for t in T_range]

    plt.figure(figsize=(9, 5))
    plt.plot(T_range, T_next_lin_T, label="Modèle Linéaire", color="red", linestyle="--", linewidth=1.5)
    plt.plot(T_range, T_next_quad_T, label="Modèle Quadratique (Effet couplé $T \\times T_{out}$)", color="blue",
             linewidth=2)
    plt.title(f"Coupe Thermique : Sensibilité à $T_{{zone}}$ ($Q$={Q_fixed_3}W, $T_{{out}}$={Tout_fixed_3}°C)")
    plt.xlabel("Température Actuelle de la Zone $T_{zone}$ [°C]")
    plt.ylabel("$T_{{next}}$ après 15 min [°C]")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Coupe_Sensibilite_Tzone.pdf", dpi=300)
    plt.savefig(f"{output_dir}/Coupe_Sensibilite_Tzone.png", dpi=300)
    plt.close('all')

