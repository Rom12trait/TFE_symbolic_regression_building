import pandas as pd
import importlib
from sklearn.linear_model import LinearRegression
from src import communs
importlib.reload(communs)
from src.function_rc import RCmodel, compute_r, slab_capacity, load_idf
from sklearn.model_selection import train_test_split
from src.linear_model import LinearRegressionModel



runfile="run_1"
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



model = LinearRegressionModel(random_state=42)

train_time = model.fit(X_train, y_train)

y_pred, test_time = model.predict(X_test)

metrics_rl = communs.compute_metrics(y_test, y_pred, train_time, test_time)


model.save_parameters("results/run_1/",
                          filename=f"{model.__class__.__name__}.json")

print("HELLO")

