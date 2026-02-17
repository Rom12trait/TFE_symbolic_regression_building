from sklearn.linear_model import LinearRegression
from src.base_model import BaseModel
import time
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

    def get_parameters_dict(self):
        return {
            "coefficients": self.model.coef_.tolist(),
            "intercept": float(self.model.intercept_),
        }