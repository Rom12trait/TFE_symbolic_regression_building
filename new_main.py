import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import importlib
from src import communs
from src import physical_models
from src import regression_models
from src import symbolic_models
importlib.reload(physical_models)
importlib.reload(communs)
importlib.reload(regression_models)
importlib.reload(symbolic_models)

from src.function_rc import load_idf
from sklearn.model_selection import train_test_split
from src.hvac_optimizer import HVACOptimizer
import pyomo.environ as pyo

from src.api_validator import EnergyPlusValidator
from src.building_model import MediumOffice


runfile="NonLinear_annee_dyn_couenne" #LinearV2_annee_classique" #LinearV2_annee_dyn
yeartype = 'cube' #ou 'exp' 'dynamique' important pour l'équation de pysr dans hvacoptimizer car j'ai dû brute force
version = 'V4'
name = "Linear"
file_paths = ["Validation_EPlus_VS_Opti", "Validation_EPlus_sans_Opti"]

#communs.agregate(runfile, "annuel_24h")
#communs.tolatex(runfile, "outputs", "annuel_24h")


#run_dir = communs.create_run_folder("run_2","results")
randomstate=42
#Charger les données
#soit airport + brussel bel ensemble ou annee dyn
#df = communs.load_data("dataset/output_energyplus/US_SF_data_energyplus_airport_15min.csv")
#df_test = communs.load_data("dataset/output_energyplus/US_SF_data_energyplus_Brussels_bel_15min.csv")
df = communs.load_data("dataset/ModeleHabitation/model_annee_dynamique.csv")
df_test = communs.load_data("dataset/ModeleHabitation/model_annee_dynamique_brussel_bel.csv")
idf = load_idf(
    "dataset/ModeleHabitation/US+SF+CZ4C+hp+slab+IECC_2024_Brussels_airport_V2420.idf",
    "C:/Users/Corentin/energyplus/Energy+.idd"
)
data_annee = communs.load_data_api("opti/EPlus_run_20_24/model_annee_classique_20_24.csv") #dataset/ModeleHabitation/anneeClassique/model_annee_classique.csv avant

X = df[["Tzone", "Tout", "Qhvac"]].values
y = df["Tzone_next"].values
#X_test = df_test[["Tzone", "Tout", "Qhvac"]].values
#y_test = df_test["Tzone_next"].values

#normalement x_val et y_val
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomstate, shuffle=False)


# 2. INSTANCIATION DES MODÈLES
models = {
    #"RC": physical_models.RCModel(R=0.008636689, C=27.14e6),
    #"Linear": regression_models.PolynomialThermalModel(degree=1),
    #"Quadratic": regression_models.RestrictedQuadraticModel(),  #regression_models.PolynomialThermalModel(degree=2),
    #"PySR": symbolic_models.PySRThermalModel(niterations=80),
    "PySR_exp": symbolic_models.PySRThermalModel(niterations=150)
}
simulations_config = [
    {
        "run_id": "pysr_square_n120_p20_1step",
        "niterations": 120,
        "populations": 20,
        "maxsize": 30,
        "parsimony": 0.0005,
        "complexity_of_constants": 2
    },

]
#%%
for config in simulations_config:
    run_id = config["run_id"]
    n_iter = config["niterations"]
    n_pop = config["populations"]
    maxsize = config["maxsize"]
    parsimony = config["parsimony"]
    complexity_of_constants = config["complexity_of_constants"]

    print(f"\n=========================================================================")
    print(f"🚀 LANCEMENT DE LA SIMULATION : {run_id}")
    print(f"Configurations : Iterations={n_iter} | Populations={n_pop}")
    print(f"=========================================================================")

    # 1. Dossier de destination unique pour centraliser TOUTES les sauvegardes de ce run
    save_dir = f"outputs/{run_id}"
    os.makedirs(save_dir, exist_ok=True)

    # 2. Instanciation dynamique avec passage du run_id
    model = symbolic_models.PySRThermalModel(niterations=n_iter,
                                             run_id=run_id,
                                             populations=n_pop,
                                             maxsize = config["maxsize"],
                                             parsimony = config["parsimony"],
                                             complexity_of_constants = config["complexity_of_constants"])

    # 3. Entraînement
    model.fit(X_train, y_train)

    # 4. Évaluation 1-step
    y_pred_1step, _ = model.predict(X_test)
    metrics_1step = communs.compute_metrics(y_test, y_pred_1step)

    # 5. Évaluation récursive 24h
    y_pred_24h, time_24h = model.simulate_yearly_24h(X)
    metrics_24h = communs.compute_metrics(X[:, 0], y_pred_24h)

    # 6. Extraction de l'expression sélectionnée
    equ = model.get_sympy_expression()
    print(f"✨ Équation Sélectionnée dénormalisée : {equ}")

    # =========================================================================
    # ARCHIVAGE SÉCURISÉ DANS LE MÊME DOSSIER (save_dir)
    # =========================================================================
    # Sauvegarde des métriques excel au même endroit
    communs.save_run_to_excel(
        filepath=f"{save_dir}/metrics_1step.xlsx",
        model_name=f"{run_id}_1step",
        metrics=metrics_1step,
        comment="prédiction sur le pas suivant"
    )

    communs.save_predictions(
        filepath=f"{save_dir}/preds_1step.xlsx",
        datetime_index=None,
        t_true=y_test,
        t_pred=y_pred_1step
    )

    communs.save_run_to_excel(
        filepath=f"{save_dir}/metrics_annuel_24h.xlsx",
        model_name=f"{run_id}_24h",
        metrics=metrics_24h,
        comment="Déroulement récursif 24h sur 365 jours"
    )

    communs.save_predictions(
        filepath=f"{save_dir}/preds_24h.xlsx",
        datetime_index=df.index,
        t_true=X[:, 0],
        t_pred=y_pred_24h
    )

    # Sauvegarde des paramètres json du modèle
    model.save_parameters(f"{save_dir}/", filename="parametres_PySR.json")

    # SAUVEGARDE TEXTE SÉPARÉE DE L'ÉQUATION RETENUE
    with open(f"{save_dir}/equation_selectionnee.txt", "w", encoding="utf-8") as f:
        f.write(f"Structure de l'equation extraite par la méthode accuracy :\n")
        f.write(f"{str(equ)}\n")

    print(f" Tout le contenu du run {run_id} a été archivé avec succès.")

print("\n Toutes les simulations automatisées sont terminées avec succès.")

#%%
selected_days, dict_days_prices, df_prix = communs.process_market_prices('dataset/prix_marché/GUI_ENERGY_PRICES_202412312300-202512312300.csv', seed = 42)
data_12days, data_annual = communs.load_data_opti_new(
    "opti/Eplus_run_20_24/model_annee_classique_20_24.csv", selected_days) #avant dataset/ModeleHabitation/anneeClassique/model_annee_classique.csv
#opti/EPlus_run_20_24/model_annee_classique_20_24.csv
# --- PARAMÈTRES PHYSIQUES ---
eta_h, eta_c = communs.calculate_average_efficiencies(data_annual) #c= 3.73 h = 1.86

# api
validator = EnergyPlusValidator(MediumOffice, data_annee)
all_validation_results = {}

for name, model in models.items():
    results_all_days = {}
    summary_data = []
    if name == "RC":
        continue  # On saute le RC pour l'optimisation
    print(f"\n--- Optimisation avec le modèle : {name} ---")
    optimizer = HVACOptimizer(thermal_model=model)

    # Dans ta boucle sur les 12 jours :
    for day in selected_days:
        prices = dict_days_prices[day]
        tout = data_12days[day]['Tout']
        t_init = data_12days[day]['Tzone_real'][0]  # Première valeur

        # Résolution automatique (détecte si Gurobi ou Ipopt est requis)
        model_resolved, _ = optimizer.solve(prices, tout, t_init, year = yeartype, mode = "nonlinear")

        # 3. Extraction des vecteurs de résultats
        p_heat_opt = [pyo.value(model_resolved.P_heating[t]) for t in model_resolved.T]
        p_cool_opt = [pyo.value(model_resolved.P_cooling[t]) for t in model_resolved.T]
        t_zone_opt = [pyo.value(model_resolved.T_zone[t]) for t in model_resolved.T_instants]
        borne_low = [pyo.value(model_resolved.low_borne[t]) for t in model_resolved.T]
        borne_high = [pyo.value(model_resolved.high_borne[t]) for t in model_resolved.T]

        total_cost = pyo.value(model_resolved.real_cost)

        step_costs = (np.array(prices) * (np.array(p_heat_opt) + np.array(p_cool_opt)) / 1000 * 0.25)
        # 4. Stockage dans le dictionnaire global
        results_all_days[day] = {
            'P_heating': p_heat_opt,
            'P_cooling': p_cool_opt,
            'T_zone': t_zone_opt,
            'T_real': data_12days[day]['Tzone_real'],
            'T_borne_low': borne_low,
            'T_borne_high': borne_high,
            'Step_Costs': list(step_costs),
            'total_Cost': total_cost,
            'Prices': prices,
            'T_min': optimizer.params["tmin"],  # Récupère 20
            'T_max': optimizer.params["tmax"],  # Récupère 24
            'P_h_max': optimizer.params["Ph_max"],
            'P_c_max': optimizer.params["Pc_max"]
        }

        # Préparation Résumé
        summary_data.append({'Day': day, 'Total_Cost_Euro': total_cost})

        print(f"✅ Terminé. Coût : {total_cost:.3f} €")


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
    #
    # --- CRÉATION DU FICHIER EXCEL MULTI-FEUILLES ---
    communs.export_opti_results_to_excel(df_summary, results_all_days,
                                         output_dir=f"opti{version}", output_path = f"Resultats_Optimisation_{name}_{runfile}.xlsx")

    # --- BOUCLE D'EXÉCUTION ---
    # On boucle sur tous les jours présents dans tes résultats (les 12 jours)
    for day in results_all_days.keys():
        print(f"Génération des graphiques pour : {day}...")
        communs.save_plot_day(day, results_all_days, output_dir=f"opti{version}/results_opti_{name}_{runfile}")

    #api

    for day in selected_days:
        # --- VALIDATION API (XPOST) ---
        t_opt = results_all_days[day]['T_zone']

        df_ep_reality = validator.run_validation(day, t_opt, name=name, output_dir=f"api{version}/{name}_{runfile}")
        # Stockage pour analyse
        all_validation_results[day] = df_ep_reality

    all_stats_list = []
    all_df_days_dict = {}
    all_stats_list_sans_opti = []
    all_df_days_dict_sans_opti = {}
    for day in results_all_days.keys():
        try:
            stats, df_cleaned = communs.analyze_variable_timestep_results(day, results_all_days, name,  csv_dir = f"api{version}/{name}_{runfile}")
            all_stats_list.append(stats)
            all_df_days_dict[day] = df_cleaned
            stats_sans_opti, df_sans_opti = communs.analyze_variable_timestep_results_sans_opti(day, results_all_days) #utilise fichier in opti/Eplus_run_20_24
            all_stats_list_sans_opti.append(stats_sans_opti)
            all_df_days_dict_sans_opti[day] = df_sans_opti
            communs.plot_comparison_results_api(day, results_all_days[day], df_cleaned, df_sans_opti, output_dir= f"api{version}/{name}_{runfile}")
            plt.close('all')
            print(f"{day} | Coût E+: {stats['Cost_Real']:.2f}€ vs Opti: {stats['Cost_Opti']:.2f}€")
        except Exception as e:
            print(f"Erreur sur {day}: {e}")
    communs.plot_global_costs_bar_chart(all_stats_list, all_stats_list_sans_opti, name=name, runfile=runfile, output_dir=f"api{version}/{name}_{runfile}")
    communs.export_validation_to_excel(all_stats_list,all_df_days_dict, output_path=f"api{version}/{name}_{runfile}/Validation_EPlus_VS_Opti.xlsx")
    communs.export_validation_to_excel(all_stats_list_sans_opti, all_df_days_dict_sans_opti, output_path=f"api{version}/{name}_{runfile}/Validation_EPlus_sans_Opti.xlsx")

    for file_path in file_paths:
            communs.to_latex(version, name, runfile, file_path)
