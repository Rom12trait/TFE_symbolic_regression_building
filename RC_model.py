import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# paramètres RC
R = 0.0008          # K/W
C = 26.82e6         # J/K
dt = 600            # s

# coefficients discrets
a = 1 - dt / (R * C)
b = dt / (R * C)
c = dt / C

# données
d = pd.read_csv("data/US_SF_data_energyplus.csv", sep=";")

# Drop the 2 design days (10-min timestep)
rows_per_day = int(24 * 60 / 10)  # 144
df = d.iloc[2 * rows_per_day :]

T = df["Tzone"].values
Tout = df["Tout"].values
Qhvac = df["Qhvac"].values


T_pred = np.zeros_like(T)
T_pred[0] = T[0]

for k in range(len(T) - 1):
    T_pred[k+1] = (
        a * T[k]
        + b * Tout[k]
        + c * Qhvac[k]
    )

rmse = np.sqrt(mean_squared_error(T[1:], T_pred[1:]))
r2 = r2_score(T[1:], T_pred[1:])

print(f"RC model RMSE = {rmse:.3f} °C")
print(f"RC model R² = {r2:.3f}")