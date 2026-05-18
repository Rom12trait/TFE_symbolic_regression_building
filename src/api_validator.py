import pandas as pd
import numpy as np
from pathlib import Path
from src.opti import EnergyPlusSimulator


class EnergyPlusValidator:
    def __init__(self, building_model_class, simulation_data):
        """
        :param building_model_class: La classe du bâtiment (ex: MediumOffice)
        :param simulation_data: Le DataFrame de l'année complète pour EnergyPlus
        """
        self.building_model_class = building_model_class
        self.simulation_data = simulation_data
        self.simulator = EnergyPlusSimulator()

    def prepare_api_input(self, day_str, t_zone_opt):
        """Convertit le vecteur optimisé en DataFrame pour l'API."""
        # On cale l'année sur 2017 pour correspondre aux fichiers météo standards
        start_dt = pd.to_datetime(day_str).replace(year=2017)
        # Générer l'index temporel (97 points car de 00:00 à 00:00 J+1)
        times = pd.date_range(start=start_dt, periods=len(t_zone_opt), freq='15min', tz='UTC')

        df = pd.DataFrame(index=times)
        df['Tin'] = t_zone_opt
        # S'assurer que le format est bien Datetime64
        df.index = pd.to_datetime(df.index)
        return df

    def run_validation(self, day, t_zone_opt, name = None, output_dir="apiV2"):
        """Lance la simulation EnergyPlus avec les consignes optimisées."""
        day_dt = pd.to_datetime(day).replace(year=2017)
        # 1. Initialiser le modèle du bâtiment pour ce jour spécifique
        building = self.building_model_class(day)
        building.idf_filepath = building.modify_idf(day_dt)
        building.simulation_exante = self.simulation_data

        # 2. Injecter les consignes optimisées (T_zone)
        df_api_input = self.prepare_api_input(day, t_zone_opt)
        for zone in building.conditioned_zone_assets:
            if zone.name == "LIVING_UNIT1":
                zone.expected_results = df_api_input
        df_final_api = {}

        # 3. Lancer EnergyPlus
        print(f"🚀 Simulation EnergyPlus pour le jour {day}...")
        df_final_api, warmup_steps = self.simulator.run_simulation(
            buildingmodel=building,
            run_period_of_interest=3,  # Paramètre de ton code initial
            callbacks=self.simulator.callback_temperature_control,
            verbose=True
        )

        # 4. Sauvegarde
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(output_dir) / f"validation_EP_{name}_{day}.csv"
        df_final_api.to_csv(save_path, sep=";")

        return df_final_api