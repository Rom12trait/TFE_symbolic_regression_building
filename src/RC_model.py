import numpy as np
import pandas as pd
from src import communs
from src.function_rc import RCmodel, compute_r, slab_capacity, load_idf

#Charger les données
df = communs.load_data("../dataset/output_energyplus/US_SF_data_energyplus.csv")
idf = load_idf(
    "../dataset/modèle habitation/US+SF+CZ4C+hp+slab+IECC_2024_Brussels_airport_V2420.idf",
    "C:/Users/Corentin/energyplus/Energy+.idd"
)

Tzone = df["Tzone"].values
Tout = df["Tout"].values
Qhvac = df["Qhvac"].values

params_rc = {
    "R": compute_r(idf),
    "C": slab_capacity(),
    "dt": 600
}



model = RCmodel(
    R=params_rc["R"], #0.0008, au début
    C=params_rc["C"], #26.82e6,
    dt=params_rc["dt"]
)

#T_pred = model.predict(
 #   Tzone=Tzone,
 #   Tout=Tout,
  #  Qhvac=Qhvac
#)

T_pred, time = communs.time_function(model.predict, Tzone, Tout, Qhvac)

#error
metrics_rc = communs.compute_metrics(Tzone, T_pred, time)



run_dir = communs.create_run_folder("run_1")

timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
communs.save_run_to_excel(
    filepath="results/run_1/RC/results_rc.xlsx",
    model_name="RC_model",
    params=params_rc,
    metrics=metrics_rc,
    comment="RC basé IDF, sans airgap"
)
communs.save_predictions(
    filepath=f"results/run_1/RC_{timestamp}.xlsx",
    datetime_index=df.index,
    t_true=Tzone,
    t_pred=T_pred
)
communs.save_rc_model(
    f"results/models/RC_{timestamp}.json",
    params_rc["R"], params_rc["C"], params_rc["dt"], model.a, model.b, model.c
)



