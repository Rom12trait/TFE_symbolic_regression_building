import pandas as pd
from pathlib import Path
# 1. Charger le fichier Excel
# Remplacez 'votre_fichier.xlsx' par le nom de votre fichier
# Si votre fichier est dans un dossier, mettez le chemin complet

project_root = Path(__file__).resolve().parents[1]
run_dir = project_root/"results"/"run_1"

file_path = run_dir / "metrics_all_models.xlsx"
df = pd.read_excel(file_path)
df_transposed= df.T
# 1. Utiliser la première ligne comme en-tête
df_transposed.columns = df_transposed.iloc[0]

# 2. Supprimer la première ligne (qui est maintenant dans l'en-tête)
# et réinitialiser l'index pour que "Models", "RC", etc. soit une colonne
df_final = df_transposed.drop(df_transposed.index[0]).reset_index()
#df_final = df_transposed.reset_index()

# 3. Optionnel : renommer la colonne d'index si nécessaire
df_final = df_final.rename(columns={'index': 'Modeles'})

# 2. Convertir en LaTeX
# index=False pour ne pas inclure la numérotation des lignes
latex_code = df_final.to_latex(index=False, caption="Mon tableau", label="tab:mon_tableau")


# 3. Sauvegarder dans un fichier .tex
with open("tableau.tex", "w", encoding="utf-8") as f:
    f.write(latex_code)

print("Conversion réussie ! Fichier 'tableau.tex' généré.")