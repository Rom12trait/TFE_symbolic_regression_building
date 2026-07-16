import numpy as np
import sympy
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from src.new_base_model import BaseModel

class PolynomialThermalModel(BaseModel):
    def __init__(self, degree=1, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.model = LinearRegression()

    def fit(self, X, y):
        X_poly = self.poly.fit_transform(X)
        start = time.perf_counter()
        self.model.fit(X_poly, y)
        self.is_fitted = True
        return time.perf_counter() - start

    def predict(self, X):
        X_poly = self.poly.transform(X)
        start = time.perf_counter()
        y_pred = self.model.predict(X_poly)
        return y_pred, time.perf_counter() - start

    def predict_step(self, T, Tout, Q):
        feat = self.poly.transform(np.array([[T, Tout, Q]]))
        return self.model.predict(feat)[0]

    def predict_24h(self, T_initial, Tout_vector, Q_vector):
        start = time.perf_counter()
        n = len(Tout_vector)
        t_preds = np.zeros(n)
        curr_t = T_initial
        for k in range(n):
            # Création du vecteur d'entrée pour le degré choisi
            features = np.array([[curr_t, Tout_vector[k], Q_vector[k]]])
            curr_t = self.model.predict(self.poly.transform(features))[0]
            t_preds[k] = curr_t
        return t_preds, time.perf_counter() - start

    def get_sympy_expression(self):
        # 1. Définir les symboles de manière isolée
        T_s, Tout_s, Q_s = sympy.symbols('T Tout Q')

        # 2. Récupérer les puissances des features (ex: [1, 0, 0] pour T, [2, 0, 0] pour T^2)
        # C'est beaucoup plus robuste que de parser des strings
        powers = self.poly.powers_
        coeffs = self.model.coef_
        intercept = self.model.intercept_

        expr = intercept

        for i in range(len(coeffs)):
            # On construit le terme : coeff * (T^p1 * Tout^p2 * Q^p3)
            p_T, p_Tout, p_Q = powers[i]
            term = (T_s ** p_T) * (Tout_s ** p_Tout) * (Q_s ** p_Q)
            expr += coeffs[i] * term

        return sympy.simplify(expr)

    def get_parameters_dict(self):
        return {"degree": self.degree, "intercept": float(self.model.intercept_), "coef": self.model.coef_.tolist()}


class RestrictedQuadraticModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = LinearRegression()

    def _engineer_features(self, X_raw):
        """X_raw: [[T, Tout, Q], ...]"""
        T = X_raw[:, 0]
        Tout = X_raw[:, 1]
        Q = X_raw[:, 2]

        # On ne crée QUE les features physiquement logiques
        X_built = np.column_stack([
            T,  # Linéaire T
            Tout,  # Linéaire Tout
            Q,  # Linéaire Q
            Tout ** 2,  # Non-linéarité météo (effet COP)
            T * Tout  # Non-linéarité des pertes thermiques par conduction
        ])
        return X_built

    def fit(self, X, y):
        X_engineered = self._engineer_features(X)
        self.model.fit(X_engineered, y)
        self.is_fitted = True

    def predict(self, X):
        """
        Prédit l'état thermique suivant.
        Prend la même structure que PolynomialThermalModel mais applique
        notre ingénierie de variables spécifique.
        """
        X_engineered = self._engineer_features(X)
        start = time.perf_counter()
        y_pred = self.model.predict(X_engineered)
        return y_pred, time.perf_counter() - start

    def predict_step(self, T, Tout, Q):
        """
        Prédit un unique pas de temps (utilisé pour la simulation 24h annuelle).
        """
        features = np.array([[T, Tout, Q]])
        X_engineered = self._engineer_features(features)
        return float(self.model.predict(X_engineered)[0])

    def get_sympy_expression(self):
        T_s, Tout_s, Q_s = sympy.symbols('T Tout Q')
        coeffs = self.model.coef_
        intercept = self.model.intercept_

        # Reconstruction stricte de l'équation validée
        expr = (intercept +
                coeffs[0] * T_s +
                coeffs[1] * Tout_s +
                coeffs[2] * Q_s +
                coeffs[3] * (Tout_s ** 2) +
                coeffs[4] * (T_s * Tout_s))
        return sympy.simplify(expr)

    def get_parameters_dict(self):
        return {
            "intercept": float(self.model.intercept_),
            "coef": self.model.coef_.tolist(),
            "variables_structure": ["T", "Tout", "Q", "Tout^2", "T*Tout"]
        }