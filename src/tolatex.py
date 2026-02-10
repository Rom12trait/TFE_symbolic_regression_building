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
df_final = df_transposed.reset_index()

# 2. Convertir en LaTeX
# index=False pour ne pas inclure la numérotation des lignes
latex_code = df_final.to_latex(index=False, caption="Mon tableau", label="tab:mon_tableau")

# 3. Sauvegarder dans un fichier .tex
with open("tableau.tex", "w") as f:
    f.write(latex_code)

print("Conversion réussie ! Fichier 'tableau.tex' généré.")