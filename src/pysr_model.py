from pysr import PySRRegressor
from sympy import false
from src.base_model import BaseModel
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
import optuna
import sympy


class PySRModel(BaseModel):

    def __init__(self, random_state=None, niterations=100, populations=10, populations_size=40, maxsize=20, parsimony= 1e-3):
        super().__init__(random_state)
        pysr_dir = Path("C:/Users/Corentin/pysr")
        pysr_dir.mkdir(parents=True, exist_ok=True)

        self.fixed_params = {
            "binary_operators": ["+", "*"],
            "unary_operators": [],
            "model_selection": "accuracy",
            "verbosity": 1,
            "elementwise_loss" : "L2DistLoss()",
            "crossover_probability": 0.04,
            "complexity_of_constants": 0.5,        # Favorise l'intercept
            "complexity_of_variables": 1,
            "constraints":{'*': (1, 1)},
            "select_k_features" : 3,
            "batching": True,
            "batch_size": 1024,
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

    def predict_24h(self, x, steps_per_day=96):
        equation_str = self.model.get_best()["equation"]
        n = len(x)
        predictions = np.zeros(n)
        f = eval(f"lambda x0, x1, x2: {equation_str.replace('x0', 'x0').replace('x1', 'x1').replace('x2', 'x2')}")
        t_out_all = x[:, 1]
        q_hvac_all = x[:, 2]
        start = time.perf_counter()
        for i in range(0, n, steps_per_day):
            t_zone_actuelle = x[i,0]
            end = min(i + steps_per_day, n)
            predictions[i] = t_zone_actuelle
            for t in range(i, end-1):
                valeur_pred = f(t_zone_actuelle, t_out_all[t], q_hvac_all[t])
                predictions[t+1]= valeur_pred
                t_zone_actuelle = valeur_pred
        elapsed = time.perf_counter() - start
        return predictions, elapsed

    def predict_24hnorm(self, x_real, scaler_X, scaler_y, steps_per_day=96):

        x_scaled = scaler_X.transform(x_real)

        # Récupération de l'équation rapide (sur données scaled)
        equation_str = self.model.get_best()["equation"]
        f_scaled = eval(f"lambda x0, x1, x2: {equation_str}")

        n = len(x_real)
        predictions_scaled = np.zeros(n)
        # Extraire les colonnes pour un accès direct (vitesse maximale)
        x1_scaled = x_scaled[:, 1]
        x2_scaled = x_scaled[:, 2]
        start = time.perf_counter()

        for i in range(0, n, steps_per_day):
            t_zone_actuelle_scaled = x_scaled[i, 0]
            predictions_scaled[i] = t_zone_actuelle_scaled

            end = min(i + steps_per_day, n)
            for t in range(i, end - 1):
                t_next_scaled = f_scaled(t_zone_actuelle_scaled, x1_scaled[t], x2_scaled[t])
                predictions_scaled[t + 1] = t_next_scaled
                t_zone_actuelle_scaled = t_next_scaled

        predictions_real = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        elapsed = time.perf_counter() - start
        return predictions_real, elapsed

    def tune_optuna(self, x_train, y_train, x_val, y_val, n_trials = 20, timeout= None):

        def objective(trial):
            params = {
                "crossover_probability": trial.suggest_int("crossover_probability", 0.02, 0.5),
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
        pysr_params = self.model.get_params()
        pysr_params_clean = {
            k: (str(v) if not isinstance(v, (int, float, str, list, dict, bool, type(None))) else v)
            for k, v in pysr_params.items()
        }
        fulldict = { **pysr_params_clean, **self.fixed_params, **bestdict}
        return fulldict

    def denormalize_pysr_equation(self, scaler_X, scaler_y):
        """
        Transforme l'équation normalisée de PySR en équation réelle.
        """
        # 1. Récupérer l'expression SymPy brute du modèle
        # (Par défaut, récupère la meilleure équation selon le score)
        expr = self.model.sympy()

        # 2. Définir les symboles (x0, x1, ...) correspondant aux colonnes de X
        n_features = scaler_X.n_features_in_
        x_symbols = [sympy.Symbol(f'x{i}') for i in range(n_features)]

        # 3. Créer le dictionnaire de substitution pour X
        # x_norm = (x_reel - mean) / std
        substitutions = {
            x_symbols[i]: (x_symbols[i] - scaler_X.mean_[i]) / scaler_X.scale_[i]
            for i in range(n_features)
        }

        # 4. Appliquer la substitution dans l'équation
        expr_with_real_x = expr.subs(substitutions)

        # 5. Dénormaliser la sortie Y
        # y_reel = (y_norm * std_y) + mean_y
        final_expr = (expr_with_real_x * scaler_y.scale_[0]) + scaler_y.mean_[0]

        # 6. Simplifier l'expression algébrique
        return sympy.simplify(final_expr)
