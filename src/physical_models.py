import numpy as np
import sympy
import time
from src.new_base_model import BaseModel

class RCModel(BaseModel):
    def __init__(self, R, C, dt=900, **kwargs):
        super().__init__(**kwargs)
        self.R = R
        self.C = C
        self.dt = dt

    @property
    def a(self): return 1 - (self.dt / (self.R * self.C))
    @property
    def b(self): return self.dt / (self.R * self.C)
    @property
    def c(self): return self.dt / self.C

    def predict(self, X):
        """X est un array [[T, Tout, Q], ...]"""
        start = time.perf_counter()
        # Prédiction 1-step basée sur les vraies valeurs T_zone (X[:,0])
        y_pred = self.a * X[:, 0] + self.b * X[:, 1] + self.c * X[:, 2]
        return y_pred, time.perf_counter() - start

    def predict_step(self, T, Tout, Q):
        return self.a * T + self.b * Tout + self.c * Q

    def predict_24h(self, T_initial, Tout_vector, Q_vector):
        """Simulation récursive sur 96 pas"""
        start = time.perf_counter()
        n = len(Tout_vector)
        t_preds = np.zeros(n)
        curr_t = T_initial
        for k in range(n):
            curr_t = self.a * curr_t + self.b * Tout_vector[k] + self.c * Q_vector[k]
            t_preds[k] = curr_t
        return t_preds, time.perf_counter() - start

    def get_sympy_expression(self):
        T, Tout, Q = sympy.symbols('T Tout Q')
        return self.a * T + self.b * Tout + self.c * Q

    def get_parameters_dict(self):
        return {"R": self.R, "C": self.C, "a": self.a, "b": self.b, "c": self.c}

    def benchmark(self, t_zone, timestep_minutes=15, day_step=4):

        steps_per_hour = 60 // timestep_minutes
        steps_per_day = 24 * steps_per_hour  # 96 si 15 min

        values = t_zone

        y_true = []
        y_pred = []

        total_days = len(values) // steps_per_day

        for day in range(0, total_days - 1, day_step):

            start_idx = day * steps_per_day
            end_idx = start_idx + steps_per_day

            # prédiction = valeurs du jour courant
            pred_day = values[start_idx:end_idx]

            # vérité = valeurs 1 pas de temps plus tard
            true_day = values[start_idx + 1: end_idx + 1]

            if len(true_day) == steps_per_day:
                y_pred.append(pred_day)
                y_true.append(true_day)

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        return y_true, y_pred