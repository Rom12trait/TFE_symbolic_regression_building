import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# 1. Charger le fichier de consigne
df_input = pd.read_csv('dataset/ModeleHabitation/consignes_hvac.csv', names=['Heating_Setpoint', 'Cooling_Setpoint'])
timestep = 15  # minutes
rows_per_day = int(24 * 60 / timestep)  # 96
design_days = 2 * rows_per_day + 1
# 2. Charger le résultat d'EnergyPlus (celui qui sort du run)
df_res = pd.read_csv('dataset/ModeleHabitation/model_annee_dynamique.csv', sep=";", skiprows=range (1, design_days))
# Nettoyage des espaces dans les noms de colonnes
df_res.columns = [c.strip() for c in df_res.columns]


plt.figure()
# Plot des consignes (Input)
# Note : si votre simulation est au pas de 10min, il faut répéter les valeurs horaires
plt.step(range(168), df_input['Heating_Setpoint'][:168], where='post',
         label='Consigne Chauffage (Input)', color='red', alpha=0.5, linestyle='--')
plt.step(range(168), df_input['Cooling_Setpoint'][:168], where='post',
         label='Consigne Cooling (Input)', color='blue', alpha=0.5, linestyle='--')

# Plot de la température réelle (Output)
# attention pas de temps
# df_res est en 15min, on prend 168*4 premières lignes
steps_per_hour = len(df_res) // 8760
res_slice = df_res.iloc[:168 * steps_per_hour]
time_axis = [i/steps_per_hour for i in range(len(res_slice))]

plt.plot(time_axis, res_slice['LIVING_UNIT1:Zone Air Temperature [C](TimeStep)'],
         label='Température Zone (Réel)', color='black', linewidth=2)
heure_semaine = np.arange(0, 169, 24)
# Mise en forme
plt.title('Validation de la Dynamique Thermique : Semaine 1')
plt.xlabel('Heures depuis le 1er Janvier')
plt.ylabel('Température [°C]')
plt.xticks(heure_semaine)
plt.grid(True, alpha=0.3)
plt.legend()
plt.xlim(0, 168) # Une semaine complète

# Ajout des noms des jours en bas pour vérifier la logique
jours = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
for i, jour in enumerate(jours):
    plt.text(i*24 + 12, plt.ylim()[0], jour, horizontalalignment='center', fontweight='bold')
plt.savefig('dynamique_thermique.pdf')
plt.show()

