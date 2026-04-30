import random
from eppy.modeleditor import IDF
from src import communs

idf_file= "dataset/ModeleHabitation/US+SF+CZ4C+hp+slab+IECC_2024_Brussels_airport_V2420.idf"
idd_file = "C:/Users/Corentin/energyplus/Energy+.idd"

IDF.setiddname(idd_file)

#%% fonctionne pas/ trop de ligne je pense
def generer_semaine_dynamique(idf):
    h_sch = idf.getobject('SCHEDULE:COMPACT', 'heating_sch')
    c_sch = idf.getobject('SCHEDULE:COMPACT', 'cooling_sch')

    # Configuration des jours
    # Structure : (Nom du jour, [Heure, Val_Heat, Heure, Val_Heat...])
    scenarios = {
        "Monday": [("Until: 24:00", 5.0)],  # Lundi : Free Floating (Chauffage très bas)
        "Tuesday": [("Until: 08:00", 15.0), ("Until: 12:00", 18.0), ("Until: 16:00", 21.0), ("Until: 24:00", 24.0)], # Mardi : Escalier montant
        "Wednesday": [("Until: 08:00", 24.0), ("Until: 12:00", 21.0), ("Until: 16:00", 18.0), ("Until: 24:00", 15.0)], # Mercredi : Escalier descendant
        "Thursday": [("Until: 10:00", 18.0), ("Until: 11:00", 28.0), ("Until: 24:00", 18.0)], # Jeudi : Pic rapide chauffage (1h)
        "Friday": [("Until: 14:00", 24.0), ("Until: 15:00", 10.0), ("Until: 24:00", 24.0)], # Vendredi : Pic rapide cooling (via setpoint bas)
        "Saturday": [("Until: 06:00", 19), ("Until: 09:00", 22), ("Until: 12:00", 19), ("Until: 15:00", 22), ("Until: 24:00", 19)],  # Samedi : Oscillation sinusoïdale (gérée après)
        "Sunday": [("Until: 12:00", 15.0), ("Until: 24:00", 25.0)]  # Dimanche : Réponse à l'échelon massif
    }

    # --- Version simplifiée et robuste pour eppy ---
    fields_h = ["Through: 12/31"]
    fields_c = ["Through: 12/31"]

    for day, values in scenarios.items():
        fields_h.append(f"For: {day}")
        fields_c.append(f"For: {day}")

        for time_str, temp in values:
            fields_h.extend([time_str, temp])
            margin = 40 if day == "Monday" else 1.0 if day == "Friday" else 5.0
            fields_c.extend([time_str, temp + margin])

    # Application des champs aux objets eppy (Field_3, Field_4, ...)
    def appliquer_champs(obj, list_values):
        # On commence à Field_1
        for i, val in enumerate(list_values, start=1):
            setattr(obj, f"Field_{i}", val)

    appliquer_champs(h_sch, fields_h)
    appliquer_champs(c_sch, fields_c)


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
        "Site Direct Solar Radiation Rate per Area"  # Utile pour voir l'impact du soleil
    ]

    for var in var_list:
        idf.newidfobject("Output:Variable",
                         Variable_Name=var,
                         Reporting_Frequency="Timestep")



# 2. Chargement du fichier IDF
idf = IDF(idf_file)

generer_semaine_dynamique(idf)
#output_idf(idf)
idf.saveas("model_dynamique2.idf")

#%%
idf2 = IDF("model_dynamique2.idf")
idf2.epw = "dataset/Meteo/Brussels.Natl.AP_BEL.epw"

#problème d'erreur avec le run à régler plus tard
idf2.run(
    readvars=True,               # Génère le fichier .csv des résultats
    output_directory="quality_data", # Dossier où seront stockés les résultats
    expandobjects=True           # Recommandé si vous utilisez des objets HVAC complexes
)



