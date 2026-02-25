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

#%% RC

model = RCmodel(
    R= 0.008636689, #compute_r(idf), 0.0009
    C= 27.14e6, #slab_capacity(), 26.82e6,
    dt=900, #15 min
    random_state= randomstate
)
#%%

T_pred, train_time_rc = communs.time_function(model.predict_free, X[:,0], X[:,1], X[:,2])

metrics_rc = communs.compute_metrics(y, T_pred, train_time_rc)


timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_rc.xlsx",
    model_name="RC_model",
    metrics=metrics_rc,
    comment="RC pas de temps"
)
communs.save_predictions(
    filepath=f"results/{runfile}/RC_predictions.xlsx",
    datetime_index=df.index,
    t_true=y,
    t_pred=T_pred
)
model.save_parameters(f"results/{runfile}")

#%%

t_pred_day = model.simulate_by_day(X[:,0], X[:,1], X[:,2])

metrics_rc_day = communs.compute_metrics(y, t_pred_day)

t_air2r2c_pred, tm_pred  = model.simulate_euler_implicite(X[:,1], X[:,2], X[0,0])
metrics_rc_2r2c = communs.compute_metrics(y, t_air2r2c_pred)

#%% benchmark 24h


y_benchmark_true, y_benchmark_pred = model.benchmark(X[:,0])


metrics_Benchmark_24h = communs.compute_metrics(y_benchmark_true, y_benchmark_pred)

communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_benchmark_rc.xlsx",
    model_name="RC_model",
    metrics=metrics_Benchmark_24h,
    comment="benchmark, step 15min, 1jour/4"
)


# 24h
model24h = RCmodel(
    R= 0.008636689, #compute_r(idf), 0.0009
    C= 27.14e6, #slab_capacity(), 26.82e6,
    dt=24*4*900, #15 min
    random_state= randomstate
)

T_pred_24, train_time_rc_24 = communs.time_function(model24h.predict_free, X[::96,0], X[::96,1], X[::96,2])

metrics_rc_24 = communs.compute_metrics(y[::96], T_pred_24, train_time_rc_24)

communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_rc.xlsx",
    model_name="RC_model",
    metrics=metrics_rc_24,
    comment="RC pas de temps 24h"
)

#%% Regression linéaire


model_rl = LinearRegressionModel(random_state=42)

train_time_rl = model_rl.fit(X_train, y_train)

y_pred, test_time_rl = model_rl.predict(X_test)

metrics_rl = communs.compute_metrics(y_test, y_pred, train_time_rl, test_time_rl)


model_rl.save_parameters(f"results/{runfile}/",
                          filename=f"{model_rl.__class__.__name__}.json")

communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_rl.xlsx",
    model_name="regression lineaire",
    metrics=metrics_rl,
    comment="rl en utilisant train_test_split, 15min"
)
communs.save_predictions(
    filepath=f"results/{runfile}/rl_predictions_.xlsx",
    datetime_index=None,
    t_true=y_test,
    t_pred=y_pred
)

#%% régression symbolique


#pysr_output_dir = Path("results/run_1", "pysr").resolve()
#pysr_output_dir.mkdir(exist_ok=True)



model_pysr = PySRModel(random_state=randomstate, niterations= 40)

train_time_pysr = model_pysr.fit_class(X_train, y_train)

y_pred_pysr, test_time_pysr = model_pysr.predict_time(X_test)

metrics_pysr = communs.compute_metrics(y_test, y_pred_pysr, train_time_pysr, test_time_pysr)

communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_pysr.xlsx",
    model_name="PySR_model",
    metrics=metrics_pysr,
    comment="PySR pour l'année airport, 15 min"
)
communs.save_predictions(
    filepath=f"results/{runfile}/pysr_predictions.xlsx",
    datetime_index=None,
    t_true=y_test,
    t_pred=y_pred_pysr
)

model_pysr.save_parameters(f"results/{runfile}/",
                          filename=f"{model_pysr.__class__.__name__}.json")







