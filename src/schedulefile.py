import pandas as pd
from eppy.modeleditor import IDF

idf_file= "../dataset/ModeleHabitation/US+SF+CZ4C+hp+slab+IECC_2024_Brussels_airport_V2420.idf"
idd_file = "C:/Users/Corentin/energyplus/Energy+.idd"

IDF.setiddname(idd_file)

# Définition de vos 7 jours (24h chacun)
# Lundi (5°C), Mardi (Escalier), etc.
lundi = [10.0] * 24
mardi = [15.0]*8 + [18.0]*4 + [21.0]*4 + [24.0]*8
mercredi = [24.0]*8 + [21.0]*4 + [18.0]*4 + [15.0]*8
jeudi = [18.0]*10 + [28.0]*1 + [18.0]*13
vendredi = [24.0]*14 + [12.0]*1 + [24.0]*9
samedi = ([19.0]*3 + [22.0]*3) * 4
dimanche = [15.0]*12 + [25.0]*12

semaine_type = lundi + mardi + mercredi + jeudi + vendredi + samedi + dimanche

# Répéter la semaine pour couvrir toute l'année (52 semaines + 1 jour)
annuel_h = (semaine_type * 53)[:8760]
annuel_c = [t + 3.0 if t > 10 else 35 for t in annuel_h] # Adaptatif pour le cooling

# Sauvegarde en CSV (sans en-tête)
df = pd.DataFrame({'heating': annuel_h, 'cooling': annuel_c})
df.to_csv('dataset/ModeleHabitation/consignes_hvac.csv', index=False, header=False)
print("Fichier 'consignes_hvac.csv' généré avec succès.")

#%%
def output_idf(idf):
    # Supprimer les anciens outputs pour éviter les doublons
    outputs = idf.idfobjects['Output:Variable']
    for o in outputs[:]:
        idf.removeidfobject(o)

        # 2. Ajouter les variables de sortie
    var_list = [
        "Site Outdoor Air Drybulb Temperature",
        "Zone Air Temperature",
        "Zone Total Internal Total Heating Rate",
        "Zone Air System Sensible Heating Rate",
        "Zone Air System Sensible Cooling Rate",
        "Site Direct Solar Radiation Rate per Area",  # Utile pour voir l'impact du soleil
        "Zone Thermostat Heating Setpoint Temperature",
        "Zone Thermostat Cooling Setpoint Temperature",
        "Zone Ventilation Sensible Heat Gain Rate",
        "Zone Ventilation Sensible Heat Loss Rate",
    ]

    for var in var_list:
        idf.newidfobject("Output:Variable",
                         Variable_Name=var,
                         Reporting_Frequency="Timestep")

idf = IDF(idf_file)

# 1. Supprimer les anciens Compacts pour éviter les conflits
for name in ['heating_sch', 'cooling_sch']:
    old = idf.getobject('SCHEDULE:COMPACT', name)
    if old: idf.removeidfobject(old)


# 2. Créer le lien vers le fichier CSV pour le chauffage
idf.newidfobject("SCHEDULE:FILE",
    Name="heating_sch",
    Schedule_Type_Limits_Name="Temperature",
    File_Name="consignes_hvac.csv", # Le fichier doit être dans le même dossier
    Column_Number=1,                # 1ère colonne du CSV
    Rows_to_Skip_at_Top=0,
    Number_of_Hours_of_Data=8760,
    Column_Separator="Comma"
)

# 3. Créer le lien pour le refroidissement
idf.newidfobject("SCHEDULE:FILE",
    Name="cooling_sch",
    Schedule_Type_Limits_Name="Temperature",
    File_Name="consignes_hvac.csv",
    Column_Number=2,                # 2ème colonne du CSV
    Rows_to_Skip_at_Top=0,
    Number_of_Hours_of_Data=8760,
    Column_Separator="Comma"
)


output_idf(idf)

idf.saveas("dataset/ModeleHabitation/anneeClassique/model_annee_classique.idf")


