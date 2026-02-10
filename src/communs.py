import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

def load_data (data_path):
    d = pd.read_csv(data_path, sep=";")
    # Drop the 2 design days (10-min timestep)
    rows_per_day = int(24 * 60 / 10)  # 144
    df = d.iloc[2 * rows_per_day:].copy()

    #décalage temporel
    df["Tzone_next"] = df["Tzone"].shift(-1)
    # Suppression de la dernière ligne (NaN)
    df = df.dropna()

    return df

def compute_metrics(y_true, y_pred, train_time = None, test_time = None):
    dt_sec = 600
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    errors = y_pred - y_true

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    max_error = np.max(np.abs(errors))
    mbe = np.mean(errors) # Montre si ton modèle : sousestime (MBE > 0) surestime (MBE < 0)

    r2 = r2_score(y_true, y_pred)
    var = np.var(errors)
    std = np.std(errors)

    cv_rmse = rmse / np.mean(y_true) * 100 #Seuils usuels : < 10 % → excellent < 15 % → acceptable


    dy_true = np.diff(y_true)/dt_sec
    dy_pred = np.diff(y_pred)/dt_sec
    rmse_derivative = np.sqrt(mean_squared_error(dy_true, dy_pred))

    metrics = {
        "MSE (°C²)": mse,
        "RMSE (°C)": rmse,
        "MAE (°C)": mae,
        "Max Error (°C)": max_error,
        "MBE / Bias (°C)": mbe,
        "R² (-)": r2,
        "CV(RMSE) (%)": cv_rmse,
        "Error Variance (°C²)": var,
        "Error Std Dev (°C)": std,
        "RMSE dT/dt (°C/s)": rmse_derivative
    }

    if train_time is not None:
        metrics["Train Time (s)"] = train_time
    if test_time is not None:
        metrics["Test Time (s)"] = test_time

    for k, v in metrics.items():
        print(f"{k:25s}: {v:.4f}")

    return metrics


def time_function(function, *args, **kwargs ):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def save_run_to_excel(
        filepath,
        model_name,
        metrics: dict,
        comment: str = ""
):

    filepath = Path(filepath)

    # Fusion paramètres + métriques
    row = {
        #"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        **metrics,
        "comment": comment
    }

    df_new = pd.DataFrame([row])


    if filepath.exists():
        df_old = pd.read_excel(filepath)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_excel(filepath, index=False)

def save_predictions(
    filepath,
    datetime_index,
    t_true,
    t_pred
):
    df = pd.DataFrame({
        "Time": datetime_index,
        "T_true": t_true,
        "T_pred": t_pred
    })

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    df.to_excel(filepath, index=False)

def save_rc_model(filepath, R, C, dt, a, b, c):
    data = {
        "model": "RC",
        "R_K_per_W": R,
        "C_J_per_K": C,
        "dt_s": dt,
        "a": a,
        "b": b,
        "c": c
    }

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def save_linear_model(filepath, coef, intercept, feature_names):
    data = {
        "model": "LinearRegression",
        "intercept": float(intercept),
        "coefficients": dict(zip(feature_names, coef))
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def save_pysr_model(filepath, model):
    with open(filepath, "w") as f:
        f.write(str(model.get_best())) # ou peut etre f.write(model.latex())



def create_run_folder(model_name, base_dir="results/"):
    run_name = f"{model_name}"
    run_path = os.path.join(base_dir, run_name)

    os.makedirs(run_path, exist_ok=True)
    return run_path

