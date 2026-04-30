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

# --- PARAMÈTRES PHYSIQUES ---
eta_h, eta_c = communs.calculate_average_efficiencies(data_annual) #c= 3.73 h = 1.86


def solve_hvac_optimization(day_str, prices_vector, Tout_vector, T_initial, steps_per_decision=96):
    """
    steps_per_decision:
        1  -> décision tous les 1/4 d'heure (original)
        4  -> décision toutes les heures (24 décisions/jour)
        96 -> décision unique pour toute la journée (z=1)
    """
    model = pyo.ConcreteModel()

    # --- INDICES ---
    model.T = pyo.RangeSet(0, 95)  # Physique (prix, Text, dynamique)
    model.T_instants = pyo.RangeSet(0, 96)  # États (Températures)

    # Calcul du nombre de blocs de décision
    num_blocks = 96 // steps_per_decision
    model.B = pyo.RangeSet(0, num_blocks - 1)  # Index des décisions réduites

    # --- PARAMÈTRES ---
    model.eta_h = pyo.Param(initialize=eta_h)
    model.eta_c = pyo.Param(initialize=eta_c)
    model.Ph_max = pyo.Param(initialize=7120.17734)
    model.Pc_max = pyo.Param(initialize=7120.17734)
    model.dt = pyo.Param(initialize=0.25)
    model.tmin = pyo.Param(initialize=20)
    model.tmax = pyo.Param(initialize=24)

    # --- VARIABLES DE DÉCISION (Indexées sur les blocs B) ---
    model.P_heating_block = pyo.Var(model.B, domain=pyo.NonNegativeReals, bounds=(0, model.Ph_max))
    model.P_cooling_block = pyo.Var(model.B, domain=pyo.NonNegativeReals, bounds=(0, model.Pc_max))
    model.z_block = pyo.Var(model.B, domain=pyo.Binary)

    # --- VARIABLE D'ÉTAT (Reste sur T_instants pour la physique) ---
    model.T_zone = pyo.Var(model.T_instants, domain=pyo.Reals, bounds=(model.tmin, model.tmax))

    # --- EXPRESSIONS (Lien entre physique T et décision B) ---
    def qhvac_rule(m, t):
        # On utilise t // steps_per_decision pour mapper le 1/4 d'heure sur le bloc
        b = t // steps_per_decision
        return (m.eta_h * m.P_heating_block[b]) - (m.eta_c * m.P_cooling_block[b])

    model.Qhvac = pyo.Expression(model.T, rule=qhvac_rule)

    def real_cost_rule(m):
        return sum(prices_vector[t] * (m.P_heating_block[t // steps_per_decision] +
                                       m.P_cooling_block[t // steps_per_decision]) / 1000 * m.dt for t in m.T)

    model.real_cost = pyo.Expression(rule=real_cost_rule)

    # --- FONCTION OBJECTIF ---
    model.cost = pyo.Objective(rule=lambda m: m.real_cost, sense=pyo.minimize)

    # --- CONTRAINTES DE CAPACITÉ / EXCLUSION (Sur les blocs B) ---
    def heat_exc_rule(m, b):
        return m.P_heating_block[b] <= m.Ph_max * m.z_block[b]

    model.heat_excl = pyo.Constraint(model.B, rule=heat_exc_rule)

    def cool_excl_rule(m, b):
        return m.P_cooling_block[b] <= m.Pc_max * (1 - m.z_block[b])

    model.cool_excl = pyo.Constraint(model.B, rule=cool_excl_rule)

    # --- DYNAMIQUE THERMIQUE (Sur T_instants) ---
    def thermal_dynamics_rule(m, t):
        if t == 0:
            return m.T_zone[0] == T_initial[0]

        a, b, c, d = 0.952113783794049, 0.0287562819434527, 0.000135946989606225, 0.709746652516824
        # Note : m.Qhvac[t-1] pointe déjà vers le bon bloc b
        return m.T_zone[t] == (a * m.T_zone[t - 1] + b * Tout_vector[t - 1] + c * m.Qhvac[t - 1] + d)

    model.dynamics = pyo.Constraint(model.T_instants, rule=thermal_dynamics_rule)

    # --- RÉSOLUTION ---
    solver = pyo.SolverFactory('gurobi')
    solver.solve(model)

    return model


# --- PARAMÈTRE DE GRANULARITÉ ---
# 1 = 15min, 4 = 1h, 96 = 24h (1 décision par jour)
steps_per_decision = 4

results_all_days = {}
summary_data = []

for day in selected_days:
    print(f"--- Optimisation en cours ({steps_per_decision} steps/bloc) : {day} ---")
    try:
        if day not in dict_days_prices or day not in data_12days:
            continue

        prices = dict_days_prices[day]
        tout = data_12days[day]['Tout']
        t_init = data_12days[day]['Tzone_real']

        # 2. Appel de l'optimiseur avec le nouveau paramètre
        model_resolved = solve_hvac_optimization(day, prices, tout, t_init, steps_per_decision=steps_per_decision)

        # 3. Extraction et RECONSTRUCTION des vecteurs 96 points
        # On fait correspondre chaque pas de temps t au bloc b correspondant
        p_heat_opt = [pyo.value(model_resolved.P_heating_block[t // steps_per_decision]) for t in range(96)]
        p_cool_opt = [pyo.value(model_resolved.P_cooling_block[t // steps_per_decision]) for t in range(96)]

        # T_zone reste sur 97 points (instants)
        t_zone_opt = [pyo.value(model_resolved.T_zone[t]) for t in model_resolved.T_instants]

        total_cost = pyo.value(model_resolved.real_cost)
        step_costs = (np.array(prices) * (np.array(p_heat_opt) + np.array(p_cool_opt)) / 1000 * 0.25)

        # 4. Stockage (Structure identique pour compatibilité avec tes fonctions)
        results_all_days[day] = {
            'P_heating': p_heat_opt,
            'P_cooling': p_cool_opt,
            'T_zone': t_zone_opt,
            'T_real': data_12days[day]['Tzone_real'],
            'Step_Costs': list(step_costs),
            'total_Cost': total_cost,
            'Prices': prices,
            'T_min': pyo.value(model_resolved.tmin),
            'T_max': pyo.value(model_resolved.tmax),
            'P_h_max': pyo.value(model_resolved.Ph_max),
            'P_c_max': pyo.value(model_resolved.Pc_max)
        }

        summary_data.append({'Day': day, 'Total_Cost_Euro': total_cost})
        print(f"✅ Terminé. Coût : {total_cost:.3f} €")

    except Exception as e:
        print(f"❌ Erreur sur le jour {day} : {e}")

# --- ANALYSE ET EXPORT ---
df_summary = pd.DataFrame(summary_data)
# Calcul de la somme totale
total_12_days = df_summary['Total_Cost_Euro'].sum()
# Création d'une ligne de résumé "Total"
# On crée un petit DataFrame pour la ligne de pied de page
df_total_row = pd.DataFrame([{'Day': 'TOTAL 12 JOURS', 'Total_Cost_Euro': total_12_days}])

# On concatène le résumé et la ligne de total
df_summary = pd.concat([df_summary, df_total_row], ignore_index=True)
print("\n--- RÉSUMÉ DES COÛTS ---")
print(df_summary)

# Export Excel
suffix = f"{steps_per_decision}steps"
communs.export_opti_results_to_excel(df_summary, results_all_days,
                                     output_path=f"opti/Resultats_Optimisation_{suffix}.xlsx")

# Génération des graphiques
for day in results_all_days.keys():
    communs.save_plot_day(day, results_all_days,
                          output_dir=f"opti/results_opti_{suffix}")