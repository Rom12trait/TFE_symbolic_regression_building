import pandas as pd
import importlib
from sklearn.linear_model import LinearRegression
from src import communs
importlib.reload(communs)
from src.function_rc import RCmodel, compute_r, slab_capacity, load_idf
from sklearn.model_selection import train_test_split
from src.pysr_model import PySRModel
import optuna
from pathlib import Path
#%%

runfile="testoptuna"
#run_dir = communs.create_run_folder("run_2","results")
randomstate=42

#Charger les données
df = communs.load_data("dataset/output_energyplus/US_SF_data_energyplus_airport_15min.csv")
df_test = communs.load_data("dataset/output_energyplus/US_SF_data_energyplus_Brussels_bel_15min.csv")

idf = load_idf(
    "dataset/modèle habitation/US+SF+CZ4C+hp+slab+IECC_2024_Brussels_airport_V2420.idf",
    "C:/Users/Corentin/energyplus/Energy+.idd"
)


X = df[["Tzone", "Tout", "Qhvac"]].values
y = df["Tzone_next"].values

X_test = df_test[["Tzone", "Tout", "Qhvac"]].values
y_test = df_test["Tzone_next"].values

X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=randomstate)

#%%
model_test = PySRModel(random_state=42)

best_param, study = model_test.tune_optuna(
    X_train, y_train,
    x_val, y_val,
    n_trials=20,
    timeout= 300
)
#%%
studyview = optuna.load_study(study_name="thermal_study", storage="sqlite:///pysr_optimization.db")
print(f"Nombre d'essais terminés : {len(studyview.trials)}")
print(f"Meilleurs hyperparamètres : {studyview.best_params}")
best_params= studyview.best_params
#%%
model = PySRModel(
    **best_params,
    random_state=randomstate
)

train_time = model.fit_class(X_train, y_train)

y_pred_pysr, test_time = model.predict_time(X_test)

metrics_pysr=communs.compute_metrics(y_test, y_pred_pysr, train_time, test_time)

communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_pysr.xlsx",
    model_name="PySR_model",
    metrics=metrics_pysr,
    comment="PySR test optuna"
)
communs.save_predictions(
    filepath=f"results/{runfile}/pysr_predictions.xlsx",
    datetime_index=None,
    t_true=y_test,
    t_pred=y_pred_pysr
)

model.save_parameters(f"results/{runfile}/",
                          filename=f"{model.__class__.__name__}.json")

print("HELLO")



