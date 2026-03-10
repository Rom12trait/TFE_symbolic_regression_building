import pandas as pd
from pathlib import Path


def aggregate_comparison():
    project_root = Path(__file__).resolve().parents[1]

    # Configuration des sources (Classique vs Dynamique)
    RUNS = {
        "Classique": project_root / "results" / "run_complet_annee_classique",
        "Dynamique": project_root / "results" / "run_complet_annee_dyn"
    }

    FILES = ["metrics_benchmark_rc.xlsx", "metrics_pysr.xlsx", "metrics_rl.xlsx", "metrics_rc.xlsx"]

    dfs_15min = []
    dfs_24h = []

    for label, run_dir in RUNS.items():
        for file_name in FILES:
            path = run_dir / file_name
            if path.exists():
                df = pd.read_excel(path)

                # Ajout de colonnes d'identification
                df["Model"] = file_name.replace("metrics_", "").replace(".xlsx", "")
                df["Dataset"] = label

                # Séparation des lignes (0 = 15min, 1 = 24h)
                # On utilise .iloc[[0]] pour garder un DataFrame d'une ligne
                dfs_15min.append(df.iloc[[0]])
                if len(df) >= 2:
                    dfs_24h.append(df.iloc[[1]])
    # Concaténation finale
    final_15min = pd.concat(dfs_15min, ignore_index=True)
    final_24h = pd.concat(dfs_24h, ignore_index=True)

    # Sauvegarde des deux fichiers Excel pour vérification
    final_15min.to_excel(project_root / "results" / "metrics_all_15min.xlsx", index=False)
    final_24h.to_excel(project_root / "results" / "metrics_all_24h.xlsx", index=False)

    return final_15min, final_24h


def export_to_latex_comparison(df, filename, caption):
    # Arrondi global
    df = df.round(3)

    df_transposed = df.T
    # 1. Utiliser la première ligne comme en-tête
    df_transposed.columns = df_transposed.iloc[0]

    # 2. Supprimer la première ligne (qui est maintenant dans l'en-tête)
    # et réinitialiser l'index pour que "Models", "RC", etc. soit une colonne
    df_final = df_transposed.drop(df_transposed.index[0]).reset_index()
    # df_final = df_transposed.reset_index()

    # 3. Optionnel : renommer la colonne d'index si nécessaire
    df_final = df_final.rename(columns={'index': 'Modeles'})

    # Export LaTeX (float_format force l'affichage homogène des décimales)
    latex_code = df_final.to_latex(
        index=False,
        caption=caption,
        label=f"tab:{filename}",
        float_format="%.3f"
    )

    output_path = Path(__file__).resolve().parents[1] / "results" / f"{filename}.tex"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"Tableau {filename}.tex généré avec succès.")


# --- EXÉCUTION ---
df_15, df_24 = aggregate_comparison()
export_to_latex_comparison(df_15, "tableau_15min", "Comparaison des modèles (Pas de temps 15 min)")
export_to_latex_comparison(df_24, "tableau_24h", "Comparaison des modèles (Déroulement 24h)")
