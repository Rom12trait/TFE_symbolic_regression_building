from pysr import PySRRegressor
from sympy import false

from src.base_model import BaseModel
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
import optuna
from itertools import product
import uuid

class PySRModel(BaseModel):

    def __init__(self, random_state=None, niterations=100, populations=10, populations_size=40, maxsize=13, parsimony= 1e-3):
        super().__init__(random_state)
        pysr_dir = Path("C:/Users/Corentin/pysr")
        pysr_dir.mkdir(parents=True, exist_ok=True)

        self.fixed_params = {
            "binary_operators": ["+", "*"],
            "unary_operators": [],
            "model_selection": "best",
            "verbosity": 1,
            "elementwise_loss" : "loss(x, y) = (x - y)^2",

        }

        self.model = PySRRegressor(
            niterations=niterations,
            random_state=random_state,
            populations=populations,
            population_size=populations_size,
            # On force un modèle linéaire
            **self.fixed_params,

            # Interdiction des non-linéarités


            parsimony= parsimony,
            maxsize=maxsize,
            output_directory= str(pysr_dir),
            temp_equation_file=false,  # Important pour le multi-threading/tuning
            delete_tempfiles=True,
            #early_stop_condition= "(loss, complexity) = (loss < 0.1) && (complexity < 10)",
        )

    def fit_class(self, x, y):
        start_pysr = time.perf_counter()
        self.model.fit(x, y)
        train_time_pysr = time.perf_counter() - start_pysr
        self.is_fitted = True
        return  train_time_pysr

    def predict_time(self, X):
        start_pysr = time.perf_counter()
        y_pred_pysr= self.model.predict(X)
        test_time_pysr = time.perf_counter() - start_pysr
        return y_pred_pysr, test_time_pysr

    def tune_optuna(self, x_train, y_train, x_val, y_val, n_trials = 20, timeout= None):

        def objective(trial):
            params = {
                "niterations": trial.suggest_int("niterations", 10, 100),
                "maxsize": trial.suggest_int("maxsize", 10, 30),
                "parsimony": trial.suggest_float("parsimony", 1e-4, 1e-2, log=True),
                "populations": trial.suggest_int("populations", 5, 20),
            }

            model = PySRRegressor(
                **self.fixed_params,
                **params,
                temp_equation_file=True
            )
            model.fit(x_train, y_train)

            y_val_pred = model.predict(x_val)

            return mean_squared_error(y_val, y_val_pred)


        study = optuna.create_study(
            study_name="thermal_study",
            storage="sqlite:///pysr_optimization.db",
            load_if_exists=True,
            direction="minimize",
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(objective, n_trials=n_trials, timeout =timeout)

        self.best_params = study.best_trial.params

        print("Best trial:")
        print(study.best_trial.params)
        print("Best validation MSE:", study.best_value)

        return self.best_params, study


    def get_parameters_dict(self):
        best = self.model.get_best()
        bestdict = {
            "equation": str(best["equation"]),
            "complexity": int(best["complexity"]),
            "loss": float(best["loss"]),
        }
        fulldict = { **self.best_params, **self.fixed_params, **bestdict}
        return fulldict
