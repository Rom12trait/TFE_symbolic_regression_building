import numpy as np
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


def analyze_variable_timestep_results(day_str, results_all_days, name, csv_dir=None):
    """
    Analyse les résultats E+ à pas de temps variables et les compare à l'opti.
    """
    csv_path = os.path.join(csv_dir, f"validation_EP_{name}_{day_str}.csv")
    df = pd.read_csv(csv_path, sep=';')

    # --- ÉTAPE DE CORRECTION DU TIMESTEP ---
    # On remplace "02/29" par "03/01" uniquement dans la colonne Timestep
    # pour que Pandas accepte de le lire en année 2017
    df['Timestep'] = df['Timestep'].str.replace("02/29", "03/01")
    # 1. Conversion du temps en datetime (en forçant l'année 2017 pour matcher l'API)
    # colonne timestep (format "MM/DD HH:MM")
    df['datetime'] = pd.to_datetime("2017/" + df['Timestep'], format="%Y/%m/%d %H:%M")

    # 2. Filtrer pour ne garder que le jour d'intérêt (exclure le warmup)
    target_date = pd.to_datetime(day_str).replace(year=2017)
    df_day = df[df['datetime'].dt.date == target_date.date()].copy()

    # 3. Calcul du dt réel en heures pour chaque ligne
    # On calcule la différence de temps avec la ligne suivante
    df_day['dt_hours'] = df_day['datetime'].diff().dt.total_seconds().shift(-1) / 3600
    # La dernière ligne n'a pas de suivante, on peut mettre 0.25 (15 min) par défaut
    df_day['dt_hours'].fillna(0.25, inplace=True)
    df_day['dt_seconds'] = df_day['dt_hours'] * 3600

    # 4. Calcul du coût réel avec prix dynamique
    # Il faut mapper le prix de l'optimisation (qui est fixe par 15min)
    # sur chaque micro-pas de temps de EnergyPlus
    res_opt = results_all_days[day_str]
    prices_96 = res_opt['Prices']  # Liste de 96 prix

    def get_price_for_time(dt):
        # Calcule l'index (0-95) dans le vecteur de prix basé sur l'heure/minute
        idx = int((dt.hour * 60 + dt.minute) // 15)
        return prices_96[min(idx, 95)]

    df_day['current_price'] = df_day['datetime'].apply(get_price_for_time)

    # --- CALCUL OPTIMISATION POUR COMPARAISON ---
    # Énergie théorique (Somme des P_watt * 0.25h / 1000)
    p_tot_opt = np.array(res_opt['P_heating']) + np.array(res_opt['P_cooling'])
    energy_opt_kwh = (p_tot_opt * 0.25).sum() / 1000
    avg_power_opt_w = p_tot_opt.mean()  # Moyenne simple des 96 points

    # Rééchantillonnage pour la température (RMSE propre)
    # On prend la moyenne de température sur chaque 15 min pour comparer à l'opti
    t_opt_96 = np.array(res_opt['T_zone'][:96])
    # On définit l'index sur le temps
    df_day = df_day.set_index('datetime')
    # On ne sélectionne QUE les colonnes de type numérique (int ou float)
    # Cela exclut automatiquement 'Timestep' (string) et 'Warmup' (bool)
    df_numeric = df_day.select_dtypes(include=[np.number])

    df_numeric['P_total_W'] = (df_numeric['Heating:Electricity'] +
                               df_numeric['Cooling:Electricity'] +
                               df_numeric['Fans:Electricity']) / 900
    df_day['step_cost_real'] = (df_numeric['P_total_W'] / 1000.0) * df_day['dt_hours'] * df_day['current_price']
    total_cost_day_ep = df_day['step_cost_real'].sum()

    # Intégration temporelle stricte pour calculer l'énergie totale du jour en Joules
    # Énergie [J] = Somme( Puissance [W] * dt [s] )
    total_energy_j = (df_numeric['P_total_W'] * df_day['dt_seconds']).sum()

    # energie totale en kwh
    total_energy_kwh = total_energy_j / 3600000
    # Puissance moyenne sur la journée [W]
    avg_power_w = total_energy_j / (24 * 3600)

    # On fait le resample sur ce DataFrame nettoyé
    df_resampled = df_numeric.resample('15min').mean()

    # Extraction des vecteurs 96 points
    p_real_96 = df_resampled['P_total_W'].iloc[:96].fillna(0).values

    # RMSE Puissance [W]
    rmse_power = np.sqrt(np.mean((p_tot_opt - p_real_96) ** 2))

    # MAPE Puissance [%]
    # Note : On ajoute un petit epsilon (1e-5) au dénominateur pour éviter la division par zéro
    # si le système est éteint (0W).
    mape_power = np.mean(np.abs((p_real_96 - p_tot_opt) / (p_real_96 + 1e-5))) * 100

    # On récupère la température
    t_real_96 = df_resampled['Zone Air Temperature,LIVING_UNIT1'].iloc[:96].values
    rmse_temp = np.sqrt(np.mean((t_opt_96 - t_real_96) ** 2))

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
                'Cout_Instant_[EUR]': values['Step_Costs'] + [np.nan]
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
    hours97 = np.linspace(0,24,97)
    hours = np.linspace(0, 24, 96)
    xticks = np.arange(0, 25, 2)
    Ph_max, Pc_max = res['P_h_max'], res['P_c_max']
    tmin, tmax = res['T_min'], res['T_max']

    # --- 1. FIGURE TEMPÉRATURES ---
    plt.figure()
    plt.plot(hours97, res['T_zone'], color='blue', label=f"T_zone {day}", linewidth=2.5)
    plt.axhline(y=tmin, color='green', linestyle='--', alpha=0.5, label=" T_min")
    plt.axhline(y=tmax, color='red', linestyle='--', alpha=0.5, label="T_max")
    plt.title(f"Analyse Thermique - {day}")
    plt.ylabel("Température [°C]")
    plt.xlabel("Heure [h]")
    plt.xticks(xticks)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Temp_Zone_{day}.png")
    plt.close()  # Ferme la figure pour libérer la RAM

    # --- 2. FIGURE PUISSANCES HVAC ---
    plt.figure() #figsize=(10, 5)
    p_net = np.array(res['P_heating']) - np.array(res['P_cooling'])
    plt.step(hours, p_net, where='post', color='orange', label="P_net (heating > 0 et cooling < 0)", linewidth=2)
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

    plt.step(hours, res['Prices'], where='post', color='blue', linewidth=1.5, label="Prix marché [€/kWh]")
    plt.step(hours, step_costs, where='post', color='black', alpha=0.7, linewidth=2, label="Coût [€/15min]")
    plt.fill_between(hours, step_costs, step='post', color='black', alpha=0.1)

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
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    # Pour l'opti (97 points de 0h à 24h)
    hours_opt = np.linspace(0, 24, 97)
    hours_p_opt = np.linspace(0, 24, 96)
    # --- Axes X ---
    time_source = df_cleaned.index if not 'datetime' in df_cleaned.columns else df_cleaned['datetime']
    # Pour EnergyPlus (on utilise les heures réelles du DataFrame nettoyé)
    hours_ep = time_source.hour + time_source.minute / 60 + time_source.second / 3600
    # Pour EnergyPlus Baseline
    time_base = df_baseline.index if not 'datetime' in df_baseline.columns else df_baseline['datetime']
    hours_ep_base = time_base.hour + time_base.minute / 60 + time_base.second / 3600

    # --- FIGURE 1 : COMPARAISON TEMPÉRATURES ---
    plt.figure()
    plt.plot(hours_opt, res_opt['T_zone'], 'r--', label="T_zone Opti (Consigne)", linewidth=2)
    plt.plot(hours_ep, df_cleaned['Zone Air Temperature,LIVING_UNIT1'], 'b-', label="T_zone EnergyPlus (Réel)", alpha=0.8)
    plt.plot(hours_ep_base, df_baseline['LIVING_UNIT1:Zone Air Temperature [C](TimeStep)'], 'k-', label="Baseline (Thermostat 20-24°C)",
             alpha=0.5)

    plt.axhline(res_opt['T_min'], color='grey', linestyle=':', alpha=0.5, label="Bornes Confort")
    plt.axhline(res_opt['T_max'], color='grey', linestyle=':', alpha=0.5)

    plt.title(f"Courbes Températures - {day_str}")
    plt.ylabel("Température [°C]")
    plt.xlabel("Heure [h]")
    plt.xticks(np.arange(0, 25, 2))
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Comp_Temp_{day_str}.pdf")
    plt.savefig(f"{output_dir}/Comp_Temp_{day_str}.png")
    plt.close()

    # --- FIGURE 2 : COMPARAISON PUISSANCES THERMIQUES (Q_hvac) ---

    # Opti : Puissance thermique théorique (96 points)
    # On utilise tes rendements : eta_h=1.86, eta_c=3.73
    q_opt = (1.86 * np.array(res_opt['P_heating'])) - (3.73 * np.array(res_opt['P_cooling']))
    hours_q_opt = np.linspace(0, 24, 96, endpoint=False)

    # E+ : Puissance thermique réelle (Sensible Heating - Sensible Cooling)
    q_real = df_cleaned['Zone Air System Sensible Heating Rate,LIVING_UNIT1'] - \
             df_cleaned['Zone Air System Sensible Cooling Rate,LIVING_UNIT1']
    q_real_base = df_baseline['LIVING_UNIT1:Zone Air System Sensible Heating Rate [W](TimeStep)'] - df_baseline[
        'LIVING_UNIT1:Zone Air System Sensible Cooling Rate [W](TimeStep)']
    plt.figure()
    plt.step(hours_q_opt, q_opt, where='post', label="Q_hvac Opti (Théorique)", color='red', alpha=0.6)
    plt.plot(hours_ep, q_real, label="Q_hvac EnergyPlus (Réel)", color='blue', alpha=0.7)
    plt.plot(hours_ep_base, q_real_base, 'k-', label="Q Baseline", alpha=0.4)

    plt.title(f"Courbes Flux Thermique - {day_str}")
    plt.ylabel("Puissance Thermique [W]")
    plt.xlabel("Heure [h]")
    plt.xticks(np.arange(0, 25, 2))
    plt.axhline(0, color='black', linewidth=0.8, alpha=0.3)
    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/Comp_Power_{day_str}.pdf")
    plt.savefig(f"{output_dir}/Comp_Power_{day_str}.png")
    plt.close()

    # --- Figure 3 Power ---
    # Opti : P_heat positif, P_cool négatif
    p_heat_opt = np.array(res_opt['P_heating'])
    p_cool_opt = -np.array(res_opt['P_cooling'])

    # EnergyPlus : Conversion Joules -> Watts moyen sur l'intervalle
    # P [W] = Energie [J] / dt [s]
    dt_sec = 900
    p_heat_ep = df_cleaned['Heating:Electricity'] / dt_sec
    p_cool_ep = -df_cleaned['Cooling:Electricity'] / dt_sec

    dt_sec_base = df_baseline['dt_hours'] * 3600
    p_heat_ep_base = df_baseline['Heating:Electricity [J](TimeStep)'] / dt_sec_base
    p_cool_ep_base = -df_baseline['Cooling:Electricity [J](TimeStep)'] / dt_sec_base
    # --- Création de la figure ---
    plt.figure(figsize=(8, 5))  # Taille adaptée pour être lisible une fois côte à côte

    # Tracés Optimisation
    plt.step(hours_p_opt, p_heat_opt, where='post', label="P_heat Opti (W)", color='red', alpha=0.5)
    plt.step(hours_p_opt, p_cool_opt, 'r--', where='post', label="P_cool Opti (W)", alpha=0.5)

    # Tracés EnergyPlus (Lignes continues)
    plt.plot(hours_ep, p_heat_ep, 'b-', label="P_heat E+ (Réel)", alpha=0.8, linewidth=1.2)
    plt.plot(hours_ep, p_cool_ep, 'b--', label="P_cool E+ (Réel)", alpha=0.8, linewidth=1.2)

    # Tracés Baseline (Gris/Noir)
    plt.plot(hours_ep_base, p_heat_ep_base, 'k-', label="P_heat Baseline", alpha=0.3)
    plt.plot(hours_ep_base, p_cool_ep_base, color='grey', label="P_cool Baseline", alpha=0.3)

    # Mise en forme
    plt.title(f"Courbes Puissances Électriques - {day_str}")
    plt.ylabel("Puissance Électrique [W] (cooling < 0)")
    plt.xlabel("Heure [h]")
    plt.xticks(np.arange(0, 25, 2))
    plt.axhline(0, color='black', linewidth=0.8, alpha=0.3)  # Ligne de zéro
    plt.legend(loc='best', fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    # Sauvegarde en PDF
    plt.savefig(f"{output_dir}/Comp_Electric_Power_{day_str}.pdf")
    plt.savefig(f"{output_dir}/Comp_Electric_Power_{day_str}.png")
    plt.close()

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

    # On remplace 24:00:00 par 00:00:00 pour que Pandas accepte la lecture
    df['Timestep_clean'] = df['Date/Time'].str.replace("24:00:00", "00:00:00")
    # On convertit en datetime
    df['datetime'] = pd.to_datetime("2017/" + df['Timestep_clean'].str.strip(), format="%Y/%m/%d %H:%M:%S")

    # Pour les lignes qui étaient à 24:00:00, on ajoute 1 jour car c'est le minuit du lendemain
    df.loc[mask_midnight, 'datetime'] = df.loc[mask_midnight, 'datetime'] + pd.Timedelta(days=1)

    # 2. Filtrer pour ne garder que le jour d'intérêt
    # Attention : le minuit (00:00:00) du jour J est maintenant inclus correctement
    #df['datetime'] = pd.to_datetime("2017/" + df['Date/Time'], format="%Y/ %m/%d %H:%M:%S")

    # 2. Filtrer pour ne garder que le jour d'intérêt (exclure le warmup)
    target_date = pd.to_datetime(day_str).replace(year=2017)
    df_day = df[df['datetime'].dt.date == target_date.date()].copy()

    # 3. Calcul du dt réel en heures pour chaque ligne
    # On calcule la différence de temps avec la ligne suivante
    df_day['dt_hours'] = df_day['datetime'].diff().dt.total_seconds().shift(-1) / 3600
    # La dernière ligne n'a pas de suivante, on peut mettre 0.25 (15 min) par défaut
    df_day['dt_hours'].fillna(0.25, inplace=True)
    df_day['dt_seconds'] = df_day['dt_hours'] * 3600

    # 4. Calcul du coût réel avec prix dynamique
    # Il faut mapper le prix de l'optimisation (qui est fixe par 15min)
    # sur chaque micro-pas de temps de EnergyPlus
    res_opt = results_all_days[day_str]
    prices_96 = res_opt['Prices']  # Liste de 96 prix

    def get_price_for_time(dt):
        # Calcule l'index (0-95) dans le vecteur de prix basé sur l'heure/minute
        idx = int((dt.hour * 60 + dt.minute) // 15)
        return prices_96[min(idx, 95)]

    df_day['current_price'] = df_day['datetime'].apply(get_price_for_time)

    # Puissance totale électrique (j)
    p_elec_j = (df_day['Heating:Electricity [J](TimeStep)'] +
                df_day['Cooling:Electricity [J](TimeStep)'] +
                df_day['Fans:Electricity [J](TimeStep) '])

    # Coût par ligne = (j / (dtsec*1000)) * dt_heures * Prix_kWh
    df_day['step_cost_real'] = (p_elec_j /(df_day['dt_seconds'] * 1000)) * df_day['dt_hours'] * df_day['current_price']
    #cout total
    total_cost_day_ep = df_day['step_cost_real'].sum()

    # Somme des énergies en Joules sur toutes les lignes du jour
    total_energy_j = (df_day['Heating:Electricity [J](TimeStep)'] +
                      df_day['Cooling:Electricity [J](TimeStep)'] +
                      df_day['Fans:Electricity [J](TimeStep) ']).sum()
    #energie totale en kwh
    total_energy_kwh = total_energy_j / 3600000
    #Puissance moyenne sur la journée [W]
    avg_power_w = total_energy_j / (24 * 3600)

    # Rééchantillonnage pour la température (RMSE propre)
    # On prend la moyenne de température sur chaque 15 min pour comparer à l'opti
    t_opt_96 = np.array(res_opt['T_zone'][:96])
    # On définit l'index sur le temps
    df_day = df_day.set_index('datetime')
    # On ne sélectionne QUE les colonnes de type numérique (int ou float)
    # Cela exclut automatiquement 'Timestep' (string) et 'Warmup' (bool)
    df_numeric = df_day.select_dtypes(include=[np.number])
    # On fait le resample sur ce DataFrame nettoyé
    df_resampled = df_numeric.resample('15min').mean()

    # On récupère la température
    t_real_96 = df_resampled['LIVING_UNIT1:Zone Air Temperature [C](TimeStep)'].iloc[:96].values

    rmse_temp = np.sqrt(np.mean((t_opt_96 - t_real_96) ** 2))
    stats = {
        'Day': day_str,
        'Cost_Real': total_cost_day_ep,
        'RMSE_Temp': rmse_temp,
        'Energy_Real_kWh': total_energy_kwh,
        'P_Avg_Real_W': avg_power_w
    }
    return stats, df_resampled
