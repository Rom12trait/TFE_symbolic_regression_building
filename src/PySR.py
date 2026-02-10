
from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from src import communs
from pathlib import Path

#Charger les données
df = communs.load_data("../dataset/output_energyplus/US_SF_data_energyplus.csv")

X = df[["Tzone", "Tout", "Qhvac"]].values
y = df["Tzone_next"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

run_dir = communs.create_run_folder("PySR")

pysr_output_dir = Path(run_dir, "pysr").resolve()
pysr_output_dir.mkdir(exist_ok=True)

modelPySR = PySRRegressor(
    niterations=80,
    populations=10,
    # On force un modèle linéaire
    binary_operators=["+", "*"],
    unary_operators=[],

    # Interdiction des non-linéarités
    constraints={
        "*": (1, 1),   # multiplication uniquement constante * variable
    },

    model_selection="best",
    elementwise_loss= "loss(x, y) = (x - y)^2",
    maxsize=13,
    verbosity=1,
    output_directory=str(pysr_output_dir),
    run_id= "run1"
)

modelPySR.fit(X_train, y_train)
#modelPySR, train_time = communs.time_function(modelPySR.fit, X_train, y_train)
print("\nÉquation trouvée :")
print(modelPySR)

print("\nMeilleure équation :")
print(modelPySR.get_best())


y_pred_test = modelPySR.predict(X_test)
#y_pred_test, test_time = communs.time_function(modelPySR.predict, X_test)
#error
metrics_pysr = communs.compute_metrics(y_test, y_pred_test)
