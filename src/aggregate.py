import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
run_dir = project_root/"results"/"run_1"
FILES = {
    "Benchmark": run_dir / "metrics_benchmark_rc.xlsx",
    "PySR": run_dir / "metrics_pysr.xlsx",
    "RL": run_dir / "metrics_rl.xlsx",
    "RC": run_dir / "metrics_rc.xlsx"
}
print("PROJECT_ROOT =", project_root)
print("RUN_DIR =", run_dir)

dfs = []

for model, path in FILES.items():
    if not path.exists():
        raise FileNotFoundError(f"❌ Fichier introuvable : {path}")
    df = pd.read_excel(path)
    df["model"] = model
    dfs.append(df)

final_df = pd.concat(dfs, ignore_index=True)

output = Path( run_dir /"metrics_all_models.xlsx")
final_df.to_excel(output, index=False)

print(f"✅ Fichier créé : {output}")