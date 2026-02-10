import pandas as pd
from sklearn.linear_model import LinearRegression
from src import communs
from sklearn.model_selection import train_test_split


#Charger les données
df = communs.load_data("../dataset/output_energyplus/US_SF_data_energyplus.csv")

#entré-sortie
X = df[["Tzone", "Tout", "Qhvac"]].values
y = df["Tzone_next"].values

params_rl = {
    "features": ["Tzone", "Tout", "Qhvac"],
    "solver": "OLS"
}


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
#model.fit(X, y)

model, train_time = communs.time_function(model.fit, X_train,y_train)
a, b, c = model.coef_
d = model.intercept_
print(f"Tzone(t+1) = {a:.4f} * Tzone(t) + {b:.4f} * Tout(t) + {c:.6f} * Q_HVAC(t) + {d:.4f}")


y_pred_test, test_time = communs.time_function(model.predict,X_test)
#error
metrics_rl = communs.compute_metrics(y_test, y_pred_test, train_time, test_time)

communs.save_run_to_excel(
    "results/results_runs.xlsx",
    "LinearRegression",
    params_rl,
    metrics_rl,
    comment="rl en utilisant train_test_split"
)
timestamp = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")

communs.save_predictions(
    filepath=f"results/predictions/rl_{timestamp}.xlsx",
    datetime_index=None,
    t_true=y_test,
    t_pred=y_pred_test
)
communs.save_linear_model(
    filepath=f"results/models/rl_{timestamp}.json",
    coef=model.coef_,
    intercept=model.intercept_,
    feature_names=params_rl["features"]
)