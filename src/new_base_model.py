import json
from pathlib import Path
from datetime import datetime
import numpy as np
import time


class BaseModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.is_fitted = False

    def get_metadata(self):
        return {
            "model_type": self.__class__.__name__,
            "random_state": self.random_state,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def save_parameters(self, directory, filename=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = filename or f"{self.__class__.__name__}_params.json"

        params = self.get_parameters_dict()
        full_data = {**self.get_metadata(), **params}

        with open(directory / filename, "w") as f:
            json.dump(full_data, f, indent=4)
        print(f" Paramètres sauvegardés dans : {directory / filename}")

    def get_parameters_dict(self):
        return {}

    def simulate_yearly_24h(self, X_full, steps_per_day=96):
        """
        Simule l'année complète par blocs de 24h.
        X_full doit contenir [Tzone, Tout, Qhvac] pour toute l'année.
        """
        n = len(X_full)
        t_preds = np.zeros(n)
        start_time = time.perf_counter()

        # On boucle jour par jour
        for day_start in range(0, n, steps_per_day):
            day_end = min(day_start + steps_per_day, n)

            # Initialisation au début de chaque jour avec la vraie température
            curr_t = X_full[day_start, 0]
            t_preds[day_start] = curr_t

            # Simulation récursive pour le reste de la journée
            for k in range(day_start, day_end - 1):
                # On utilise la méthode predict_step qui sera propre à chaque modèle
                curr_t = self.predict_step(curr_t, X_full[k, 1], X_full[k, 2])
                t_preds[k + 1] = curr_t

        elapsed = time.perf_counter() - start_time
        return t_preds, elapsed