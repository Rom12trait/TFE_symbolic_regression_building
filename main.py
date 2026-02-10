import pandas as pd
from sklearn.linear_model import LinearRegression
from src import communs
from src.function_rc import RCmodel, compute_r, slab_capacity, load_idf
from sklearn.model_selection import train_test_split
from pysr import PySRRegressor
from pathlib import Path
import time


runfile="run_1"
run_dir = communs.create_run_folder(f"{runfile}","results")

#Charger les données
df = communs.load_data("dataset/output_energyplus/US_SF_data_energyplus.csv")
idf = load_idf(
    "dataset/modèle habitation/US+SF+CZ4C+hp+slab+IECC_2024_Brussels_airport_V2420.idf",
    "C:/Users/Corentin/energyplus/Energy+.idd"
)


X = df[["Tzone", "Tout", "Qhvac"]].values
y = df["Tzone_next"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#%% RC
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

T_pred, train_time_rc = communs.time_function(model.predict, X[:,0], X[:,1], X[:,2])

metrics_rc = communs.compute_metrics(X[:,0], T_pred, train_time_rc)


timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_rc.xlsx",
    model_name="RC_model",
    metrics=metrics_rc,
    comment="RC basé IDF, sans airgap"
)
communs.save_predictions(
    filepath=f"results/{runfile}/RC_predictions_{timestamp}.xlsx",
    datetime_index=df.index,
    t_true=X[:,0],
    t_pred=T_pred
)
communs.save_rc_model(
    f"results/{runfile}/RC_model_{timestamp}.json",
    params_rc["R"], params_rc["C"], params_rc["dt"], model.a, model.b, model.c
)

#%% benchmark 24h

dt = 600  # 10 min
horizon_hours = 24
N = int(horizon_hours * 3600 / dt)

T_pred_24h = model.predict_free(
    Tzone=X[:,0],
    Tout=X[:N, 1],
    Qhvac=X[:N, 2]
)

T_true_24h = y[:N]

metrics_Benchmark_24h = communs.compute_metrics(T_true_24h, T_pred_24h)

timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_benchmark_rc.xlsx",
    model_name="RC_model",
    metrics=metrics_rc,
    comment="RC basé IDF, sans airgap"
)

#%% Regression linéaire


params_rl = {
    "features": ["Tzone", "Tout", "Qhvac"],
    "solver": "OLS"
}

model = LinearRegression()
model, train_time = communs.time_function(model.fit, X_train,y_train)
a, b, c = model.coef_
d = model.intercept_
print(f"Tzone(t+1) = {a:.4f} * Tzone(t) + {b:.4f} * Tout(t) + {c:.6f} * Q_HVAC(t) + {d:.4f}")

y_pred_test, test_time = communs.time_function(model.predict,X_test)
#error
metrics_rl = communs.compute_metrics(y_test, y_pred_test, train_time, test_time)


communs.save_run_to_excel(
    f"results/{runfile}/metrics_rl.xlsx",
    "LinearRegression",
    metrics_rl,
    comment="rl en utilisant train_test_split"
)
communs.save_predictions(
    filepath=f"results/{runfile}/rl_predictions_{timestamp}.xlsx",
    datetime_index=None,
    t_true=y_test,
    t_pred=y_pred_test
)
communs.save_linear_model(
    filepath=f"results/{runfile}/rl_model{timestamp}.json",
    coef=model.coef_,
    intercept=model.intercept_,
    feature_names=params_rl["features"]
)

#%% régression symbolique
params_pysr = {
    "niterations": 100,
    "populations": 10,
    "binary_operators":["+", "*"],
    "unary_operators":[],
    "constraints":{
        "*": (1, 1),   # multiplication uniquement constante * variable
    },
    "model_selection":"best",
    "elementwise_loss":"loss(x, y) = (x - y)^2",
    "maxsize":13,
    "verbosity":1,
}

pysr_output_dir = Path(run_dir, "pysr").resolve()
pysr_output_dir.mkdir(exist_ok=True)

modelPySR = PySRRegressor(
    niterations=100,
    populations=10,
    # On force un modèle linéaire
    binary_operators=["+", "*"],
    unary_operators=[],

    # Interdiction des non-linéarités
    constraints={
        "*": (1, 1),   # multiplication uniquement constante * variable
    },

    model_selection="best",
    elementwise_loss="loss(x, y) = (x - y)^2",
    maxsize=13,
    verbosity=1,
    output_directory=str(pysr_output_dir),
)
start_pysr = time.perf_counter()
modelPySR.fit(X_train, y_train)
train_time_pysr = time.perf_counter() - start_pysr

#modelPySR, train_time_pysr = communs.time_function(modelPySR.fit, X_train, y_train)
print("\nÉquation trouvée :")
print(modelPySR)

print("\nMeilleure équation :")
print(modelPySR.get_best())


y_pred_pysr_test= modelPySR.predict(X_test)
#y_pred_test, test_time_pysr = communs.time_function(modelPySR.predict, X_test)
#error
metrics_pysr = communs.compute_metrics(y_test, y_pred_pysr_test, train_time_pysr)
#%%
communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_pysr.xlsx",
    model_name="PySR_model",
    metrics=metrics_pysr,
    comment="PySR pour l'année airport"
)
#%%
communs.save_predictions(
    filepath=f"results/{runfile}/pysr_predictions.xlsx",
    datetime_index=None,
    t_true=y_test,
    t_pred=y_pred_pysr_test
)
communs.save_pysr_model(
    filepath=f"results/{runfile}/pysr_model.json",
    model=modelPySR
)

#%%








