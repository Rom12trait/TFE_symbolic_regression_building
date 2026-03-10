from sklearn.linear_model import LinearRegression
from src.base_model import BaseModel
import time
import numpy as np
class LinearRegressionModel(BaseModel):

    def __init__(self, random_state=None):
        super().__init__(random_state)
        self.model = LinearRegression()

    def fit(self, X, y):
        self.set_seed()
        start = time.perf_counter()
        self.model.fit(X, y)
        elapsed = time.perf_counter() - start
        self.is_fitted = True
        return elapsed

    def predict(self, X):
        start = time.perf_counter()
        pred =self.model.predict(X)
        elapsed = time.perf_counter() - start
        return pred, elapsed

    def predict24hour(self, t_zone, t_out, q_hvac, steps_per_day=96):
        n = len(t_zone)
        t_pred = np.zeros(n)

        a = self.model.coef_[0]
        b = self.model.coef_[1]
        c = self.model.coef_[2]
        d = self.model.intercept_
        print(a)
        print(b)
        print(c)
        print(d)
        start_time = time.perf_counter()

        for i in range(0, n, steps_per_day):
            end = min(i + steps_per_day, n)
            t_pred[i] = t_zone[i]

            for k in range(i, end-1):
                t_pred[k+1] = a*t_pred[k] + b* t_out[k] + c* q_hvac[k] + d

        elapsed = time.perf_counter() - start_time
        return t_pred, elapsed

    def get_parameters_dict(self):
        return {
            "coefficients": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
        }