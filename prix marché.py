import pandas as pd
import importlib
import numpy as np
from src import communs
import matplotlib.pyplot as plt
import pyomo.environ as pyo
importlib.reload(communs)

selected_days, dict_days_prices, df_prix = communs.process_market_prices('dataset/prix_marché/GUI_ENERGY_PRICES_202412312300-202512312300.csv', seed = 42)
data_12days, data_annual = communs.load_data_opti_new("dataset/modèle habitation/anneeClassique/model_annee_classique.csv", selected_days)

# --- PARAMÈTRES PHYSIQUES ---
eta_h, eta_c = communs.calculate_average_efficiencies(data_annual) #c= 3.73 h = 1.86


def solve_hvac_optimization(day_str, prices_vector, Tout_vector, T_initial, Tset_heat, Tset_cool):
    """+ Pfans_vector[t]
    day_str: '2025-01-12'
    prices_vector: array de 96 prix (€/kWh)
    Tout_vector: array de 96 températures extérieures (EnergyPlus)
    """
    model = pyo.ConcreteModel()
    # 1. Indices (0 à 95 pour les 96 quartiers d'heure)
    model.T = pyo.RangeSet(0, 95)

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
    model.T_zone = pyo.Var(model.T, domain=pyo.Reals, bounds = (model.tmin, model.tmax))
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
        # On arrête la règle à t=94 pour que t+1 ne dépasse pas 95
        #if t >= 95:
        #    return pyo.Constraint.Skip

        # Équation : T_future (t+1) = f(T_actuelle, Tout_actuelle, Qhvac_actuelle)
        # On utilise tes coefficients a, b, c, d
        a, b, c, d = 0.952113783794049, 0.0287562819434527, 0.000135946989606225, 0.709746652516824

        return m.T_zone[t] == (a * m.T_zone[t-1] + b * Tout_vector[t-1] + c * m.Qhvac[t-1] + d)

    model.dynamics = pyo.Constraint(model.T, rule=thermal_dynamics_rule)

    # --- RÉSOLUTION ---
    solver = pyo.SolverFactory('gurobi')
    solver.solve(model)

    return model


# --- BOUCLE D'EXÉCUTION DES 12 JOURS ---
results_all_days = {}

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
        Tset_heat = data_12days[day]['Tset_heat']
        Tset_cool = data_12days[day]['Tset_cool']

        # 2. Appel de l'optimiseur
        model_resolved = solve_hvac_optimization(day, prices, tout, t_init, Tset_heat, Tset_cool)

        # 3. Extraction des vecteurs de résultats
        p_heat_opt = [pyo.value(model_resolved.P_heating[t]) for t in model_resolved.T]
        p_cool_opt = [pyo.value(model_resolved.P_cooling[t]) for t in model_resolved.T]
        t_zone_opt = [pyo.value(model_resolved.T_zone[t]) for t in model_resolved.T]
        total_cost = pyo.value(model_resolved.real_cost)

        # 4. Stockage dans le dictionnaire global
        results_all_days[day] = {
            'P_heating': p_heat_opt,
            'P_cooling': p_cool_opt,
            'T_zone': t_zone_opt,
            'T_real': data_12days[day]['Tzone_real'],
            'Tset_heat' : data_12days[day]['Tset_heat'],
            'Tset_cool' : data_12days[day]['Tset_cool'],
            'Cost': total_cost,
            'Prices': prices
        }

        print(f"✅ Terminé. Coût optimisé : {total_cost:.3f} €")

    except Exception as e:
        print(f"❌ Erreur sur le jour {day} : {e}")

# --- ANALYSE GLOBALE ---
total_study_cost = sum(res['Cost'] for res in results_all_days.values())
print(f"\nCoût total pour les 12 jours sélectionnés : {total_study_cost:.2f} €")
print("hello")
# Exemple pour exporter un jour spécifique vers un CSV pour analyse
# pd.DataFrame(results_all_days[selected_days[0]]).to_csv("resultat_optimisation_J1.csv")


# 1. Créer le DataFrame détaillé (96 points par jour)
all_dfs = []
summary_data = []

for day, values in results_all_days.items():
    # On crée le DataFrame pour les profils temporels du jour J
    df_temp = pd.DataFrame({
        'P_heating': values['P_heating'],
        'P_cooling': values['P_cooling'],
        'T_zone': values['T_zone'],
        'Prices': values['Prices']
    })
    df_temp['Day'] = day
    df_temp['Timestep'] = range(96)
    all_dfs.append(df_temp)

    # On stocke le coût à part dans une liste pour le résumé
    summary_data.append({'Day': day, 'Total_Cost_Euro': values['Cost']})

# DataFrame avec tous les détails (365 * 96 lignes potentiellement)
df_final = pd.concat(all_dfs, ignore_index=True)

# DataFrame de résumé (12 lignes, une par jour)
df_summary = pd.DataFrame(summary_data)
Results = pd.DataFrame(results_all_days)
data_days = pd.DataFrame(data_12days)
# --- AFFICHAGE ET EXPORT ---
print("\n--- RÉSUMÉ DES COÛTS PAR JOUR ---")
print(df_summary)
print("hello")


def plot_selected_days(results_all_days, data_12days, days_to_plot):
    """
    days_to_plot: ex ['2025-01-21', '2025-06-08']
    """
    Ph_max = 7120.17734
    Pc_max = 7120.17734
    tmin = 20
    tmax=24
    hours = np.linspace(0, 24, 96, endpoint=False)
    xticks = np.arange(0, 25, 2)

    # Couleurs contrastées : Bleu (Hiver/Froid) et Orange (Eté/Chaud)
    colors = ['blue', 'orange']

    # --- FIGURE 1 : TEMPÉRATURES ---
    plt.figure()
    for i, day in enumerate(days_to_plot):
        if day in results_all_days:
            res, data = results_all_days[day], data_12days[day]
            c = colors[i % len(colors)]
            # Zone réelle optimisée
            plt.plot(hours, res['T_zone'], color=c, label=f"T_zone {day}", linewidth=2.5)
            # Bornes (Setpoints) avec styles distincts pour ne pas confondre

    plt.axhline(y=tmin, color='blue', linestyle='--', alpha=0.3, label="Température minimale")
    plt.axhline(y=tmax, color='red', linestyle='--', alpha=0.3, label="Température maximale")
    plt.title("Analyse Thermique : Évolution de la Température de Zone")
    plt.ylabel("Température [°C]")
    plt.xlabel("Heure [h]")
    plt.xticks(xticks)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # --- FIGURE 2 : PUISSANCES HVAC ---
    plt.figure()
    for i, day in enumerate(days_to_plot):
        if day in results_all_days:
            res = results_all_days[day]
            c = colors[i % len(colors)]
            p_net = np.array(res['P_heating']) - np.array(res['P_cooling'])
            plt.step(hours, p_net, where='post', color=c, label=f"P_net {day}", linewidth=2)

    plt.axhline(y=Ph_max, color='red', linestyle='--', alpha=0.3, label="P_max Chauffage")
    plt.axhline(y=-Pc_max, color='blue', linestyle='--', alpha=0.3, label="P_max Refroidissement")
    plt.title("Profil de Puissance Électrique HVAC")
    plt.ylabel("Puissance [W] (Chaud > 0 / Froid < 0)")
    plt.xlabel("Heure [h]")
    plt.xticks(xticks)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    # --- FIGURE 3 : PRIX ET COÛT CUMULÉ (Axe Simple) ---
    plt.figure()
    bar_width = 0.15
    for i, day in enumerate(days_to_plot):
        if day in results_all_days:
            res = results_all_days[day]

            # 1. Calcul du coût réel payé à l'instant t (€ pour 15 min)
            # On récupère Pfans depuis les données d'EnergyPlus
            p_tot = np.array(res['P_heating']) + np.array(res['P_cooling'])
            step_costs = (res['Prices'] * p_tot / 1000 * 0.25)

            # 2. Affichage du Prix ENTSO-E (Courbe en escalier - Pointillés)
            plt.step(hours, res['Prices'], where='post', color='blue', linestyle='--',
                     alpha=0.5, label=f"Prix ENTSO-E {day} [€/kWh]")

            # 3. Affichage du Coût par pas de temps (Barres)
            # On décale légèrement les barres du 2ème jour pour la visibilité
            offset = i * bar_width
            plt.bar(hours + offset, step_costs, width=bar_width, color='black',
                    alpha=0.8, label=f"Coût HVAC {day} [€/15min]", align='edge')

    plt.axhline(0, color='black', linewidth=0.8, alpha=0.5)  # Ligne de zéro
    plt.title("Analyse Économique : Signal de Prix et coût par Quart d'Heure")
    plt.ylabel("Valeur (€ ou €/kWh)")
    plt.xlabel("Heure [h]")
    plt.xticks(xticks)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()

# --- EXEMPLE D'APPEL ---
jours_interessants = ['2025-01-21'] # Un jour cher et un jour avec prix négatifs
plot_selected_days(results_all_days, data_12days, jours_interessants)
