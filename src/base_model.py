import json
from pathlib import Path
from datetime import datetime
import numpy as np

class BaseModel:

    def __init__(self, random_state=None):
        self.random_state = random_state
        self.is_fitted = False

    def set_seed(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def get_metadata(self):
        return {
            "model_type": self.__class__.__name__,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def save_parameters(self, directory, filename="model_parameters.json"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        params = self.get_parameters_dict()
        metadata = self.get_metadata()

        full_dict = {**metadata, **params}

        with open(directory / filename, "w") as f:
            json.dump(full_dict, f, indent=4)

        print(f"Parameters saved in {directory / filename}")