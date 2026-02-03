

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from pysr import PySRRegressor


#Charger les données
d = pd.read_csv("data/US_SF_data_energyplus.csv", sep=";")

# Drop the 2 design days (10-min timestep)
rows_per_day = int(24 * 60 / 10)  # 144
df = d.iloc[2 * rows_per_day :].copy()

df["Tzone_next"] = df["Tzone"].shift(-1)
df = df.dropna()

X = df[["Tzone", "Tout", "Qhvac"]].values
y = df["Tzone_next"].values

model = PySRRegressor(
    niterations=100,
    populations=10,
    # On force un modèle linéaire
    binary_operators=["+", "*"],
    unary_operators=[],

    # Interdiction des non-linéarités
    constraints={
        "*": (1, 1),   # multiplication uniquement constante * variable
    },

    model_selection="best",
    loss="loss(x, y) = (x - y)^2",
    maxsize=13,
    verbosity=1,
)

model.fit(X, y)

print("\nÉquation trouvée :")
print(model)

print("\nMeilleure équation :")
print(model.get_best())

#error
y_pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE = {rmse:.3f} °C")
print(f"R² = {r2:.4f}")
