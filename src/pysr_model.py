from pysr import PySRRegressor
from sympy import false

from src.base_model import BaseModel
import time
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error
from itertools import product
import uuid

class PySRModel(BaseModel):

    def __init__(self, random_state=None, niterations=100, populations=10, populations_size=40, maxsize=13, parsimony= 1e-3):
        super().__init__(random_state)
        pysr_dir = Path("C:/Users/Corentin/pysr")
        pysr_dir.mkdir(parents=True, exist_ok=True)

        self.model = PySRRegressor(
            niterations=niterations,
            random_state=random_state,
            verbosity=1,
            populations=populations,
            population_size=populations_size,
            # On force un modèle linéaire
            binary_operators=["+", "*"],
            unary_operators=[],

            # Interdiction des non-linéarités
            constraints={
                "*": (1, 1),  # multiplication uniquement constante * variable
            },
            model_selection="best",
            elementwise_loss="loss(x, y) = (x - y)^2",
            parsimony= parsimony,
            maxsize=maxsize,
            output_directory= str(pysr_dir),
            temp_equation_file=False,  # Important pour le multi-threading/tuning
            delete_tempfiles=True,
        )

    def fit_class(self, X, y):
        start_pysr = time.perf_counter()
        self.model.fit(X, y)
        train_time_pysr = time.perf_counter() - start_pysr
        self.is_fitted = True
        return  train_time_pysr

    def predict_time(self, X):
        start_pysr = time.perf_counter()
        y_pred_pysr= self.model.predict(X)
        test_time_pysr = time.perf_counter() - start_pysr
        return y_pred_pysr, test_time_pysr

    def get_parameters_dict(self):
        best = self.model.get_best()

        return {
            "equation": str(best["equation"]),
            "complexity": int(best["complexity"]),
            "loss": float(best["loss"]),
        }
