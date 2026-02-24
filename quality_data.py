import random
from eppy.modeleditor import IDF
from src import communs

idf_file= "dataset/modèle habitation/US+SF+CZ4C+hp+slab+IECC_2024_Brussels_airport_V2420.idf"
idd_file = "C:/Users/Corentin/energyplus/Energy+.idd"

IDF.setiddname(idd_file)

#%%
def generer_planning_agite(idf, base_heat=20.0, amplitude=4.0):
    # On récupère le premier objet correspondant au nom
    h_sch = idf.getobject('SCHEDULE:COMPACT', 'heating_sch')
    c_sch = idf.getobject('SCHEDULE:COMPACT', 'cooling_sch')

    if h_sch:
        # Valeurs pour 4 paliers (on assigne CHAQUE champ séparément)
        val1, val2, val3, val4 = [base_heat + random.uniform(-amplitude, amplitude) for _ in range(4)]

        # CHAUFFAGE (Structure: Until, Value, Until, Value...)
        h_sch.Field_3 = "Until: 06:00"
        h_sch.Field_4 = val1
        h_sch.Field_5 = "Until: 12:00"
        h_sch.Field_6 = val2
        h_sch.Field_7 = "Until: 18:00"
        h_sch.Field_8 = val3
        h_sch.Field_9 = "Until: 24:00"
        h_sch.Field_10 = val4

    if c_sch:
        # REFROIDISSEMENT (On garde une marge de sécurité de 5°C)
        c_sch.Field_3, c_sch.Field_5, c_sch.Field_7, c_sch.Field_9 = "Until: 06:00", "Until: 12:00", "Until: 18:00", "Until: 24:00"
        c_sch.Field_4 = val1 + 5.0
        c_sch.Field_6 = val2 + 5.0
        c_sch.Field_8 = val3 + 5.0
        c_sch.Field_10 = val4 + 5.0

    print(
        f"Planning {h_sch.Name} généré avec des variations entre {base_heat - amplitude}°C et {base_heat + amplitude}°C")


# 2. Chargement du fichier IDF
idf = IDF(idf_file)

generer_planning_agite(idf)
idf.saveas("model_dynamique.idf")

#%%
idf2 = IDF("model_dynamique.idf")
idf2.epw = "dataset/Météo/Brussels.Natl.AP_BEL.epw"

#problème d'erreur avec le run à régler plus tard
#idf2.run(
    #readvars=True,               # Génère le fichier .csv des résultats
    #output_directory="quality_data", # Dossier où seront stockés les résultats
    #expandobjects=True           # Recommandé si vous utilisez des objets HVAC complexes
#)



