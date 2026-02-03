
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


#Charger les données
d = pd.read_csv("data/US_SF_data_energyplus.csv", sep=";")


# Drop the 2 design days (10-min timestep)
rows_per_day = int(24 * 60 / 10)  # 144
df = d.iloc[2 * rows_per_day :]

#Colonnes attendues (adapter à csv)
#'Tzone', 'Tout', 'Qhvac'

#Décalage temporel
df["Tzone_next"] = df["Tzone"].shift(-1)

#Suppression de la dernière ligne (NaN)
df = df.dropna()

#entré-sortie
X = df[["Tzone", "Tout", "Qhvac"]].values
y = df["Tzone_next"].values


model = LinearRegression()
model.fit(X, y)

a, b, c = model.coef_
d = model.intercept_

print(f"Tzone(t+1) = {a:.4f} * Tzone(t) + {b:.4f} * Tout(t) + {c:.6f} * Q_HVAC(t) + {d:.4f}")


#error
y_pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print(f"RMSE = {rmse:.3f} °C")
print(f"R² = {r2:.4f}")


plt.figure()
plt.plot(y, label="EnergyPlus")
plt.legend()
plt.xlabel("Temps")
plt.ylabel("Température zone [°C]")
plt.figure()
plt.plot(y_pred, label="Modèle linéaire")
plt.legend()
plt.xlabel("Temps")
plt.ylabel("Température zone [°C]")
plt.show()