import numpy as np
import importlib
from src import communs, pysr_model, linear_model, function_rc
importlib.reload(communs)
importlib.reload(pysr_model)
importlib.reload(linear_model)
importlib.reload(function_rc)
from src.function_rc import RCmodel, load_idf
from sklearn.model_selection import train_test_split
from src.linear_model import LinearRegressionModel
from src.pysr_model import PySRModel


runfile="run_complet_annee_dyn"
#run_dir = communs.create_run_folder("run_2","results")
randomstate=42

#Charger les données
#soit airport + brussel bel ensemble ou annee dyn
#df = communs.load_data("dataset/output_energyplus/US_SF_data_energyplus_airport_15min.csv")
#df_test = communs.load_data("dataset/output_energyplus/US_SF_data_energyplus_Brussels_bel_15min.csv")
df = communs.load_data("dataset/modèle habitation/model_annee_dynamique.csv")
df_test = communs.load_data("dataset/modèle habitation/model_annee_dynamique_brussel_bel.csv")
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


#X_test = df_test[["Tzone", "Tout", "Qhvac"]].values
#y_test = df_test["Tzone_next"].values

#normalement x_val et y_val
X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=randomstate)

test1 = X[:,0]
test2 = X[:,1]
Test3 = X[:,2]

#%% RC
model = RCmodel(
    res= 0.008636689, #compute_r(idf), 0.0009
    capa= 27.14e6, #slab_capacity(), 26.82e6,
    dt=900, #15 min
    random_state= randomstate
)

# pas de temps 15min
T_pred, train_time_rc = model.predict(X[:,0], X[:,1], X[:,2])
metrics_rc = communs.compute_metrics(test1, T_pred, train_time_rc)

communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_rc.xlsx",
    model_name="RC_model",
    metrics=metrics_rc,
    comment="RC pas de temps 15min"
)
communs.save_predictions(
    filepath=f"results/{runfile}/RC_predictions.xlsx",
    datetime_index=df.index,
    t_true=test1,
    t_pred=T_pred
)
model.save_parameters(f"results/{runfile}")

#%%

t_pred_day, train_time_rc_24 = model.simulate_by_day(X[:,0], X[:,1], X[:,2])
metrics_rc_24 = communs.compute_metrics(test1, t_pred_day, train_time_rc_24)

communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_rc.xlsx",
    model_name="RC_model",
    metrics=metrics_rc_24,
    comment="RC déroulement 24h"
)

#%% benchmark 24h okay

y_benchmark_true, y_benchmark_pred = model.benchmark(X[:,0])
metrics_Benchmark_24h = communs.compute_metrics(y_benchmark_true, y_benchmark_pred)

communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_benchmark_rc.xlsx",
    model_name="RC_model",
    metrics=metrics_Benchmark_24h,
    comment="benchmark, step 15min, 1jour/4"
)

#%% Regression linéaire


model_rl = LinearRegressionModel(random_state=42)

train_time_rl = model_rl.fit(X_train, y_train)

y_pred, test_time_rl = model_rl.predict(x_val)

metrics_rl = communs.compute_metrics(y_val, y_pred, train_time_rl, test_time_rl)


model_rl.save_parameters(f"results/{runfile}/",
                          filename=f"{model_rl.__class__.__name__}.json")

communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_rl.xlsx",
    model_name="regression linéaire",
    metrics=metrics_rl,
    comment="rl pas de temps 15min"
)
communs.save_predictions(
    filepath=f"results/{runfile}/rl_predictions_.xlsx",
    datetime_index=None,
    t_true=y_val,
    t_pred=y_pred
)

# rl 24h

t_pred_24h, test_time_24h = model_rl.predict24hour(X[:,0], X[:,1], X[:,2])
metrics_rl_24h = communs.compute_metrics(X[:,0], t_pred_24h, None, test_time_24h)

communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_rl.xlsx",
    model_name="regression linéaire",
    metrics=metrics_rl_24h,
    comment="rl déroulement 24H"
)
communs.save_predictions(
    filepath=f"results/{runfile}/rl_24h_predictions_.xlsx",
    datetime_index=None,
    t_true=X[:,0],
    t_pred=t_pred_24h
)

#%% régression symbolique


#pysr_output_dir = Path("results/run_1", "pysr").resolve()
#pysr_output_dir.mkdir(exist_ok=True)



model_pysr = PySRModel(random_state=randomstate, niterations= 40)

train_time_pysr = model_pysr.fit_class(X_train, y_train)

y_pred_pysr, test_time_pysr = model_pysr.predict_time(x_val)

metrics_pysr = communs.compute_metrics(y_val, y_pred_pysr, train_time_pysr, test_time_pysr)

y_pred_pysr_24h, test_time_pysr_24h = model_pysr.predict_24h(X)
metrics_pysr_24h = communs.compute_metrics(X[:,0], y_pred_pysr_24h, None, test_time_pysr_24h)


communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_pysr.xlsx",
    model_name="PySR_model",
    metrics=metrics_pysr,
    comment="PySR pour l'année airport, 15 min"
)
communs.save_predictions(
    filepath=f"results/{runfile}/pysr_predictions.xlsx",
    datetime_index=None,
    t_true=y_val,
    t_pred=y_pred_pysr
)

model_pysr.save_parameters(f"results/{runfile}/",
                          filename=f"{model_pysr.__class__.__name__}.json")

communs.save_run_to_excel(
    filepath=f"results/{runfile}/metrics_pysr.xlsx",
    model_name="PySR_model",
    metrics=metrics_pysr_24h,
    comment="PySR déroulement 24h"
)
communs.save_predictions(
    filepath=f"results/{runfile}/pysr_predictions_24h.xlsx",
    datetime_index=None,
    t_true=X[:,0],
    t_pred=y_pred_pysr_24h
)

#%%
communs.agregate(runfile)
#%%
communs.tolatex(runfile)





