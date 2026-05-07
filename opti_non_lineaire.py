#%%
import pandas as pd
import importlib
import numpy as np
from src import communs
import pyomo.environ as pyo
from datetime import datetime, timedelta
import zoneinfo
importlib.reload(communs)
from src.opti import EnergyPlusSimulator
from src.building_model import BuildingModel, MediumOffice

selected_days, dict_days_prices, df_prix = communs.process_market_prices('dataset/prix_marché/GUI_ENERGY_PRICES_202412312300-202512312300.csv', seed = 42)
data_12days, data_annual = communs.load_data_opti_new(
    "dataset/ModeleHabitation/anneeClassique/model_annee_classique.csv", selected_days)
#opti/EPlus_run_20_24/model_annee_classique_20_24.csv
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
    #model.z = pyo.Var(model.T, domain = pyo.Binary) # sur les 96
    model.z = pyo.Var(domain=pyo.Binary) # 1 mode sur la journée

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
        return m.P_heating[t] <= m.Ph_max * m.z    #Z[t] SI 96 choix de mode
    model.heat_excl = pyo.Constraint(model.T, rule=heat_exc_rule)
    def cool_excl_rule(m,t):
        return m.P_cooling[t] <= m.Pc_max * (1-m.z) #Z[t] SI 96 choix de mode
    model.cool_excl = pyo.Constraint(model.T, rule=cool_excl_rule)

    def thermal_dynamics_nonlinear_rule(m, t):
        if t == 0:
            return m.T_zone[0] == T_initial[0]

        # Tes coefficients extraits

        #c = { annee classique 22 24
        #    'intercept': -0.6525, 'T': 1.0811e+00, 'Text': -3.0963e-02, 'Q': -1.5546e-04,
        #    'T2': -2.3372e-03, 'T_Text': 1.4482e-03, 'T_Q': 7.0463e-06,
        #    'Text2': -4.0213e-05, 'Text_Q': -2.9369e-07, 'Q2': 2.6866e-10
        #}
        c = { #annee dyn
            'intercept': 0.2750,
            'T': 1.0211e+00,
            'Text': -1.6858e-02,
            'Q': -2.7234e-04,
            'T2': -1.8108e-03,
            'T_Text': 1.1038e-03,
            'T_Q': 1.3927e-05,
            'Text2': 1.5066e-04,
            'Text_Q': -1.5541e-06,
            'Q2': -5.1683e-09
        }

        # Raccourcis pour la lisibilité
        Tk = m.T_zone[t - 1]
        Tx = Tout_vector[t - 1]
        Qk = m.Qhvac[t - 1]

        # L'équation quadratique complète
        return m.T_zone[t] == (
                c['intercept'] +
                c['T'] * Tk + c['Text'] * Tx + c['Q'] * Qk +
                c['T2'] * (Tk ** 2) +
                c['T_Text'] * (Tk * Tx) +
                c['T_Q'] * (Tk * Qk) +
                c['Text2'] * (Tx ** 2) +
                c['Text_Q'] * (Tx * Qk) +
                c['Q2'] * (Qk ** 2)
        )
    model.dynamics = pyo.Constraint(model.T_instants, rule=thermal_dynamics_nonlinear_rule)

    # --- RÉSOLUTION ---
    solver = pyo.SolverFactory('gurobi')
    solver.options['NonConvex'] = 2
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
total_12_days = df_summary['Total_Cost_Euro'].sum()
df_total_row = pd.DataFrame([{'Day': 'TOTAL 12 JOURS', 'Total_Cost_Euro': total_12_days}])

# On concatène le résumé et la ligne de total
df_summary = pd.concat([df_summary, df_total_row], ignore_index=True)
print("\n--- RÉSUMÉ DES COÛTS PAR JOUR ---")
print(df_summary)
#%%
# --- CRÉATION DU FICHIER EXCEL MULTI-FEUILLES ---
communs.export_opti_results_to_excel(df_summary, results_all_days, output_path ="opti/Resultats_Optimisation_equ_reg_lin.xlsx")

# --- BOUCLE D'EXÉCUTION ---
# On boucle sur tous les jours présents dans tes résultats (les 12 jours)
for day in results_all_days.keys():
   print(f"Génération des graphiques pour : {day}...")
   communs.save_plot_day(day, results_all_days, output_dir="opti/results_opti_equ_reg_lin")

