import numpy as np
import pandas as pd
import importlib
from src import communs, pysr_model
importlib.reload(communs)
importlib.reload(pysr_model)
from src.function_rc import RCmodel, compute_r, slab_capacity, load_idf
from sklearn.model_selection import train_test_split
from src.linear_model import LinearRegressionModel
from src.pysr_model import PySRModel


runfile="run_latest"
#run_dir = communs.create_run_folder("run_2","results")
randomstate=42

#Charger les données
df = communs.load_data("dataset/output_energyplus/US_SF_data_energyplus_airport_15min.csv")
df_test = communs.load_data("dataset/output_energyplus/US_SF_data_energyplus_Brussels_bel_15min.csv")
#df_quality = communs.load_data("dataset/generate_quality_data/model_dynamique.csv")
idf = load_idf(
    "dataset/modèle habitation/US+SF+CZ4C+hp+slab+IECC_2024_Brussels_airport_V2420.idf",
    "C:/Users/Corentin/energyplus/Energy+.idd"
)


X = df[["Tzone", "Tout", "Qhvac"]].values
y = df["Tzone_next"].values

y_hour =[]
k=0
for i in range(1, len(X[:,0]), 4):

    y_hour.append(X[i,0])
y_hour=np.array(y_hour)


X_test = df_test[["Tzone", "Tout", "Qhvac"]].values
y_test = df_test["Tzone_next"].values

X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=randomstate)

test1 = X[:,0]
test2 = X[:,1]
Test3 = X[:,2]


resultats_RC_15min = pd.read_csv("results/run_latest/RC_predictions.xlsx")
