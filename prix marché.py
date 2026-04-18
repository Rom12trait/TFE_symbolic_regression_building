#%%
import pandas as pd
import importlib
import numpy as np
from src import communs
import matplotlib.pyplot as plt
import pyomo.environ as pyo
from datetime import datetime, timedelta
import zoneinfo
from src.communs import load_data_api
importlib.reload(communs)
from src.opti import EnergyPlusSimulator
from src.building_model import BuildingModel, MediumOffice

selected_days, dict_days_prices, df_prix = communs.process_market_prices('dataset/prix_marché/GUI_ENERGY_PRICES_202412312300-202512312300.csv', seed = 42)
data_12days, data_annual = communs.load_data_opti_new(
    "dataset/ModeleHabitation/anneeClassique/model_annee_classique.csv", selected_days)

# --- PARAMÈTRES PHYSIQUES ---
eta_h, eta_c = communs.calculate_average_efficiencies(data_annual) #c= 3.73 h = 1.86


def solve_hvac_optimization(day_str, prices_vector, Tout_vector, T_initial):
    """+ Pfans_vector[t]
    day_str: '2025-01-12'
    prices_vector: array de 96 prix (€/kWh)
    Tout_vector: array de 96 températures extérieures (EnergyPlus)
    """
    model = pyo.ConcreteModel()
    # 1. Indices (0 à 95 pour les 96 quartiers d'heure)
    model.T = pyo.RangeSet(0, 95)
    model.T_instants = pyo.RangeSet(0, 96)  # Pour T_zone (97 points : de T0 à T96)

    # --- Paramètres ---
    model.eta_h = pyo.Param(initialize=eta_h)
    model.eta_c = pyo.Param(initialize=eta_c)
    model.Ph_max = pyo.Param(initialize=7120.17734)
    model.Pc_max = pyo.Param(initialize=7120.17734)
    model.dt = pyo.Param(initialize=0.25)
    model.tmin = pyo.Param(initialize=20)
    model.tmax = pyo.Param(initialize=24)

    # --- VARIABLES DE DÉCISION ---
    model.P_heating = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, model.Ph_max))
    model.P_cooling = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, model.Pc_max))
    model.T_zone = pyo.Var(model.T_instants, domain=pyo.Reals, bounds = (model.tmin, model.tmax))
    # --- variable binaire pour non-simultanéité ---
    model.z = pyo.Var(model.T, domain = pyo.Binary)

    # --- expression
    def phvac_total_rule(m, t):
        return m.P_heating[t] + m.P_cooling[t] #+ Pfans_vector[t]
    model.Phvac = pyo.Expression(model.T, rule=phvac_total_rule)

    def real_cost_rule(m):
        return sum(prices_vector[t] * (m.P_heating[t] + m.P_cooling[t]) / 1000 * m.dt for t in m.T)

    model.real_cost = pyo.Expression(rule=real_cost_rule)

    def qhvac_rule(m, t):
        return (m.eta_h * m.P_heating[t]) - (m.eta_c * m.P_cooling[t])
    model.Qhvac = pyo.Expression(model.T, rule=qhvac_rule)

    # --- FONCTION OBJECTIF (Coût total €) ---
    def objective_rule(m):
        # Coût = Prix * (P_heat + P_cool + P_fans) * 0.25
        # il y a toujours un Pfans
        return sum(prices_vector[t] * (m.P_heating[t] + m.P_cooling[t])/1000 * m.dt for t in m.T)

    model.cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    # --- contraintes ---
    #def comfort_rule(m, t):
     #T_min[t] <= T_zone[t] <= T_max[t]
     #return Tset_heat[t], m.T_zone[t], Tset_cool[t]
    #model.comfort = pyo.Constraint(model.T, rule=comfort_rule)

    def heat_exc_rule(m, t):
        return m.P_heating[t] <= m.Ph_max * m.z[t]
    model.heat_excl = pyo.Constraint(model.T, rule=heat_exc_rule)
    def cool_excl_rule(m,t):
        return m.P_cooling[t] <= m.Pc_max * (1-m.z[t])
    model.cool_excl = pyo.Constraint(model.T, rule=cool_excl_rule)

    def thermal_dynamics_rule(m, t):

        # Condition initiale à minuit (t=0)
        if t ==0:
            return m.T_zone[0] == T_initial[0]
        #if t >= 95:
        #    return pyo.Constraint.Skip

        # On utilise tes coefficients a, b, c, d
        a, b, c, d = 0.952113783794049, 0.0287562819434527, 0.000135946989606225, 0.709746652516824

        return m.T_zone[t] == (a * m.T_zone[t-1] + b * Tout_vector[t-1] + c * m.Qhvac[t-1] + d)

    model.dynamics = pyo.Constraint(model.T_instants, rule=thermal_dynamics_rule)

    # --- RÉSOLUTION ---
    solver = pyo.SolverFactory('gurobi')
    solver.solve(model)

    return model


# --- BOUCLE D'EXÉCUTION DES 12 JOURS ---
results_all_days = {}
summary_data = []

for day in selected_days:
    print(f"--- Optimisation en cours pour le jour : {day} ---")
    try:
        # 1. Vérification de la présence des données
        if day not in dict_days_prices or day not in data_12days:
            print(f" Données manquantes pour le jour {day}. Passage au suivant.")
            continue
        # 1. Extraction des données préparées
        prices = dict_days_prices[day]
        tout = data_12days[day]['Tout']
        t_init = data_12days[day]['Tzone_real']
        pfans = data_12days[day]['Pfans']  # Assure-toi que Pfans est bien dans ton dictionnaire commun


        # 2. Appel de l'optimiseur
        model_resolved = solve_hvac_optimization(day, prices, tout, t_init)

        # 3. Extraction des vecteurs de résultats
        p_heat_opt = [pyo.value(model_resolved.P_heating[t]) for t in model_resolved.T]
        p_cool_opt = [pyo.value(model_resolved.P_cooling[t]) for t in model_resolved.T]
        t_zone_opt = [pyo.value(model_resolved.T_zone[t]) for t in model_resolved.T_instants]
        total_cost = pyo.value(model_resolved.real_cost)

        step_costs = (np.array(prices) * (np.array(p_heat_opt) + np.array(p_cool_opt)) / 1000 * 0.25)
        # 4. Stockage dans le dictionnaire global
        results_all_days[day] = {
            'P_heating': p_heat_opt,
            'P_cooling': p_cool_opt,
            'T_zone': t_zone_opt,
            'T_real': data_12days[day]['Tzone_real'],
            'Step_Costs': list(step_costs),
            'total_Cost': total_cost,
            'Prices': prices,
            'T_min': pyo.value(model_resolved.tmin),  # Récupère 20
            'T_max': pyo.value(model_resolved.tmax),  # Récupère 24
            'P_h_max': pyo.value(model_resolved.Ph_max),
            'P_c_max': pyo.value(model_resolved.Pc_max)
        }

        #Préparation Résumé
        summary_data.append({'Day': day, 'Total_Cost_Euro': total_cost})

        print(f"✅ Terminé. Coût : {total_cost:.3f} €")

    except Exception as e:
        print(f"❌ Erreur sur le jour {day} : {e}")

# --- ANALYSE GLOBALE ---
Results = pd.DataFrame(results_all_days)
data_days = pd.DataFrame(data_12days)
df_summary = pd.DataFrame(summary_data)
print("\n--- RÉSUMÉ DES COÛTS PAR JOUR ---")
print(df_summary)
print(f"\nCoût total pour les 12 jours sélectionnés : {df_summary['Total_Cost_Euro'].sum():.2f} €")

# --- 2. CRÉATION DU FICHIER EXCEL MULTI-FEUILLES ---

def export_opti_results_to_excel(df_summary, results_all_days, output_path ="opti/Resultats_Optimisation_Detaille.xlsx"):
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
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

#export_opti_results_to_excel(df_summary, results_all_days)

def save_plot_day(day, results_all_days, output_dir="opti/results_image"):
    """
    Génère et sauvegarde les 3 graphiques pour un jour spécifique.
    """

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


# --- BOUCLE D'EXÉCUTION ---
# On boucle sur tous les jours présents dans tes résultats (les 12 jours)
#for day in results_all_days.keys():
 #   print(f"Génération des graphiques pour : {day}...")
  #  save_plot_day(day, results_all_days)

#%%
def prepare_for_api(day_str, t_zone_opt):
    """
    Convertit le vecteur de 96 points en DataFrame temporel pour l'API.
    """
    # Création de l'index sur 97 points (00:00 à 00:00 J+1)
    # On utilise l'argument 'day_str' pour caler la date
    start_dt = pd.to_datetime(day_str).replace(year=2017)

    # Générer 97 timestamps espacés de 15 min
    times = pd.date_range(start=start_dt, periods=len(t_opt), freq='15min', tz= 'UTC')

    # Création du DataFrame que l'API va interpoler
    df = pd.DataFrame(index=times)
    df['Tin'] = t_zone_opt


    # S'assurer que le format est bien Datetime64
    df.index = pd.to_datetime(df.index)

    return df

data_annee = communs.load_data_api("dataset/ModeleHabitation/anneeClassique/model_annee_classique.csv")

for day in selected_days:
    if day in results_all_days:
        day_dt = pd.to_datetime(day).replace(year=2017)
        building_model = MediumOffice(day)
        building_model.idf_filepath = building_model.modify_idf(day_dt)
        # 1. Récupérer les T_zone optimisées
        t_opt = results_all_days[day]['T_zone']

        # 2. Préparer le DataFrame de consigne
        df_api_input = prepare_for_api(day, t_opt)

        # 3. Injecter dans ton objet BuildingModel
        # Il faut que l'objet 'z' dans 'conditioned_zone_assets' reçoive ce DF
        for zone in building_model.conditioned_zone_assets:
            print(zone.name)
            if zone.name == "LIVINGUNIT":
                zone.expected_results = df_api_input
        building_model.simulation_exante = data_annee
        # 4. Configurer le simulateur
        simulator = EnergyPlusSimulator()

        # 5. Lancer la simulation de validation
        # run_period_of_interest est souvent 1 (dépend de ton IDF)
        df_final_api, warmup_steps = simulator.run_simulation(
            buildingmodel=building_model,
            run_period_of_interest=3,
            callbacks=simulator.callback_temperature_control,
            verbose=True
        )

        # 6. Sauvegarder les vrais résultats EnergyPlus
        df_final_api.to_csv(f"opti/bat/validation_EP_{day}.csv", sep=";")


#%%traitement données

