import pandas as pd
import importlib

import random
from src import communs
import pyomo.environ as pyo
importlib.reload(communs)

# 1. Charger le fichier
df = pd.read_csv('dataset/prix_marché/GUI_ENERGY_PRICES_202412312300-202512312300.csv')

# 2. Nettoyage de la colonne temporelle
# On prend la partie gauche avant le " - "
df['datetime_str'] = df['MTU (CET/CEST)'].str.split(' - ').str[0]

# On retire les mentions (CET) ou (CEST) qui font planter le to_datetime
df['datetime_str'] = df['datetime_str'].str.replace(' (CET)', '', regex=False)
df['datetime_str'] = df['datetime_str'].str.replace(' (CEST)', '', regex=False)

# Conversion avec format mixte pour être tranquille
df['datetime'] = pd.to_datetime(df['datetime_str'], dayfirst=True, errors='coerce')

# --- SOLUTION À L'ERREUR DES DOUBLONS ---
# On ne garde qu'une seule entrée par horodatage s'il y a des répétitions
df = df.drop_duplicates(subset=['datetime'])
df.set_index('datetime', inplace=True)
df = df.sort_index()

# 3. Conversion d'unité : EUR/MWh -> EUR/kWh
# Division par 1000
df['price_eur_kwh'] = df['Day-ahead Price (EUR/MWh)'] / 1000

# 4. Interpolation
prices_series = df['price_eur_kwh'].copy()

# On identifie la période "paliers" (avant le 1er octobre 2025)
mask_paliers = prices_series.index < '2025-10-01'
# Pour cette période, on ne garde que les points "pile à l'heure" (:00)
# pour forcer l'interpolation à créer une rampe entre H et H+1
prices_to_interp = prices_series.copy()
prices_to_interp.loc[mask_paliers & (prices_to_interp.index.minute != 0)] = None

# On lance l'interpolation sur tout le dataset
# - Avant oct : il comblera les 'None' par une rampe entre les heures.
# - Après oct : il ne touchera à rien car il n'y a pas de 'None'.
df['prices_15min'] = prices_to_interp.interpolate(method='linear')
#prices_15min = df['price_eur_kwh'].resample('15T').interpolate(method='linear')

# 5. Sélection des 12 jours (Reproductible)
random.seed(42)  # Fixe le hasard
#selected_days = []
#for month in range(1, 13):
    # On choisit un jour entre le 1 et le 28
 #   day = random.randint(1, 28)
#    date_obj = pd.Timestamp(year=2025, month=month, day=day)
 #   selected_days.append(date_obj)
selected_days = [f"2025-{m:02d}-{random.randint(1, 28):02d}" for m in range(1, 13)]

#print("Jours sélectionnés :", [d.strftime('%Y-%m-%d') for d in selected_days])

# 5. Stockage des vecteurs (96 points par jour)
dict_days_prices = {}
for day in selected_days:
    start_ts = pd.Timestamp(day + " 00:00:00")
    end_ts = pd.Timestamp(day + " 23:45:00")
    try:
        data = df['prices_15min'].loc[start_ts:end_ts]
        if len(data) >= 96:
            dict_days_prices[day] = data.iloc[:96].values
            print(f"✅ Jour {day} : {len(data.iloc[:96])} points prêts.")
    except KeyError:
        print(f"❌ Jour {day} : Données manquantes.")




data_12days = communs.load_data_opti("dataset/modèle habitation/model_annee_dynamique.csv", selected_days)



# --- PARAMÈTRES PHYSIQUES (À adapter selon tes calculs EnergyPlus) ---
eta_h = 1.48    # Efficacité chauffage (Q_heat / P_heat)
eta_c = 3.0      # Efficacité refroidissement (Q_cool / P_cool)
P_h_max = 5000   # Watts
P_c_max = 5000   # Watts
T_min = 18     # Confort min
T_max = 21  # Confort max
dt = 0.25        # 15 minutes = 0.25 heure

def solve_hvac_optimization(day_str, prices_vector, Tout_vector, T_initial):
    """+ Pfans_vector[t]
    day_str: '2025-01-12'
    prices_vector: array de 96 prix (€/kWh)
    Tout_vector: array de 96 températures extérieures (EnergyPlus)
    """
    model = pyo.ConcreteModel()
    # 1. Indices (0 à 95 pour les 96 quartiers d'heure)
    model.T = pyo.RangeSet(0, 95)

    # --- VARIABLES DE DÉCISION ---
    model.P_heating = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, P_h_max))
    model.P_cooling = pyo.Var(model.T, domain=pyo.NonNegativeReals, bounds=(0, P_c_max))
    model.T_zone = pyo.Var(model.T, domain=pyo.Reals, bounds=(T_min, T_max))


    # --- expression
    def phvac_total_rule(m, t):
        return m.P_heating[t] + m.P_cooling[t] #+ Pfans_vector[t]
    model.Phvac = pyo.Expression(model.T, rule=phvac_total_rule)

    def qhvac_rule(m, t):
        return (eta_h * m.P_heating[t]) - (eta_c * m.P_cooling[t])
    model.Qhvac = pyo.Expression(model.T, rule=qhvac_rule)

    # --- FONCTION OBJECTIF (Coût total €) ---
    def objective_rule(m):
        # Coût = Prix * (P_heat + P_cool + P_fans) * 0.25
        # il y a toujours un Pfans
        return sum(prices_vector[t] * (m.P_heating[t] + m.P_cooling[t])/1000 * dt for t in m.T)

    model.cost = pyo.Objective(rule=objective_rule, sense=pyo.minimize)


    #contraintes
    # 2. Dynamique Thermique (Modèle PySR / Linéaire)
    def thermal_dynamics_rule(m, t):
        # Condition initiale à minuit (t=0)
        if t == 0:
            return m.T_zone[0] == T_initial

        # On arrête la règle à t=94 pour que t+1 ne dépasse pas 95
        if t >= 95:
            return pyo.Constraint.Skip

        # Équation : T_future (t+1) = f(T_actuelle, Tout_actuelle, Qhvac_actuelle)
        # On utilise tes coefficients a, b, c, d
        a, b, c, d = 0.952113783794049, 0.0287562819434527, 0.000135946989606225, 0.709746652516824

        return m.T_zone[t + 1] == (a * m.T_zone[t] + b * Tout_vector[t] + c * m.Qhvac[t] + d)

    model.dynamics = pyo.Constraint(model.T, rule=thermal_dynamics_rule)

    # --- RÉSOLUTION ---
    # Si équation est linéaire, glpk est parfait.
    #ipopt pour non linéaire
    solver = pyo.SolverFactory('glpk')

    solver.solve(model)

    return model


# --- BOUCLE D'EXÉCUTION DES 12 JOURS ---

results_all_days = {}

for day in selected_days:
    print(f"--- Optimisation en cours pour le jour : {day} ---")
    try:
        # 1. Vérification de la présence des données
        if day not in dict_days_prices or day not in data_12days:
            print(f"⚠️ Données manquantes pour le jour {day}. Passage au suivant.")
            continue
        # 1. Extraction des données préparées
        prices = dict_days_prices[day]
        tout = data_12days[day]['Tout']
        t_init = data_12days[day]['Tzone_init']
        pfans = data_12days[day]['Pfans']  # Assure-toi que Pfans est bien dans ton dictionnaire commun

        # 2. Appel de l'optimiseur
        model_resolved = solve_hvac_optimization(day, prices, tout, t_init)

        # 3. Extraction des vecteurs de résultats
        p_heat_opt = [pyo.value(model_resolved.P_heating[t]) for t in model_resolved.T]
        p_cool_opt = [pyo.value(model_resolved.P_cooling[t]) for t in model_resolved.T]
        t_zone_opt = [pyo.value(model_resolved.T_zone[t]) for t in model_resolved.T]
        total_cost = pyo.value(model_resolved.cost)

        # 4. Stockage dans le dictionnaire global
        results_all_days[day] = {
            'P_heating': p_heat_opt,
            'P_cooling': p_cool_opt,
            'T_zone': t_zone_opt,
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
data_days = pd.DataFrame(data_12days)
# --- AFFICHAGE ET EXPORT ---
print("\n--- RÉSUMÉ DES COÛTS PAR JOUR ---")
print(df_summary)
print("hello")
