import numpy as np
import sympy
import time
from pysr import PySRRegressor
from sklearn.preprocessing import StandardScaler
from src.new_base_model import BaseModel


class PySRThermalModel(BaseModel):
    def __init__(self, niterations=80, **kwargs):
        super().__init__(**kwargs)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.degree = 1
        self.model = PySRRegressor(
            niterations=niterations,
            populations=10,
            population_size=40,
            parsimony=1e-3,
            maxsize=20,
            binary_operators=["+", "*"],
            unary_operators= [],
            model_selection= "accuracy",
            verbosity= 1,
            elementwise_loss= "L2DistLoss()",
            crossover_probability= 0.04,
            complexity_of_constants= 0.8,  # Favorise l'intercept
            complexity_of_variables= 1,
            constraints= {'*': (1, 1)},
            select_k_features= 3,
            batching= True,
            batch_size= 1024,
            **kwargs
        )

    def fit(self, X, y):
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        start = time.perf_counter()
        self.model.fit(X_scaled, y_scaled)
        self.is_fitted = True
        return time.perf_counter() - start

    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        start = time.perf_counter()
        y_scaled = self.model.predict(X_scaled)
        y_real = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        return y_real, time.perf_counter() - start

    def predict_step(self, T, Tout, Q):
        if not hasattr(self, '_fast_fn'):
            expr = self.get_sympy_expression()
            vars = sympy.symbols('T Tout Q')
            # On utilise cse=True pour accélérer le calcul si l'expression est complexe
            self._fast_fn = sympy.lambdify(vars, expr, modules='numpy', cse=True)

        # On appelle la fonction et on force la conversion en float
        # On ajoute [0] ou float() au cas où lambdify retourne un array
        val = self._fast_fn(T, Tout, Q)
        return float(np.array(val).item())

    def predict_24h(self, T_initial, Tout_vector, Q_vector):
        start = time.perf_counter()
        # Utilisation de lambdify pour transformer l'expression SymPy en fonction rapide
        expr = self.get_sympy_expression()
        T, Tout, Q = sympy.symbols('T Tout Q')
        f_fast = sympy.lambdify((T, Tout, Q), expr, 'numpy')

        n = len(Tout_vector)
        t_preds = np.zeros(n)
        curr_t = T_initial
        for k in range(n):
            curr_t = f_fast(curr_t, Tout_vector[k], Q_vector[k])
            t_preds[k] = curr_t
        return t_preds, time.perf_counter() - start

    def get_sympy_expression(self):
        raw_expr = self.model.sympy()
        T, Tout, Q = sympy.symbols('T Tout Q')
        # Dénormalisation symbolique
        subs = {
            sympy.Symbol('T'): (T - self.scaler_X.mean_[0]) / self.scaler_X.scale_[0],
            sympy.Symbol('Tout'): (Tout - self.scaler_X.mean_[1]) / self.scaler_X.scale_[1],
            sympy.Symbol('Q'): (Q - self.scaler_X.mean_[2]) / self.scaler_X.scale_[2]
        }
        expr_denorm = raw_expr.subs(subs)
        final_expr = (expr_denorm * self.scaler_y.scale_[0]) + self.scaler_y.mean_[0]
        return sympy.simplify(final_expr)

    def get_parameters_dict(self):
        best = self.model.get_best()
        return {"equation": str(best['equation']), "loss": float(best['loss'])}

    def simulate_yearly_24h(self, X_full, steps_per_day=96):
        x_scaled = self.scaler_X.transform(X_full)

        # Récupération de l'équation rapide (sur données scaled)
        equation_str = self.model.get_best()["equation"]
        f_scaled = eval(f"lambda x0, x1, x2: {equation_str}")

        n = len(X_full)
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

        predictions_real = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
        elapsed = time.perf_counter() - start
        return predictions_real, elapsed
