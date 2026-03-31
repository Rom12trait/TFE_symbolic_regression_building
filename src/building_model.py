import numpy as np
import pandas as pd
from pathlib import Path
import re
import sys
import tempfile

class BuildingModel:
    """
       BuildingModel is a class which contains the necessary information about an EnergyPlus existing model to simulate it.
       self.name: the name of the model
       self.idf_path: the path to the idf file
       self.weather_path: the path to the weather file
       self.output_path: the path to the folder where to save the output files of EnergyPlus
       self.zone_names: a list of the zone_names names
       self.acus: dictionary of the AirConditioningUnit names as values and the floor as keys
       self.zones_df: a dataframe with the name of the zones, the floor, the ACU and whether the zone is a plenum
       self.zone_assets: a pd.Series of Zone objects with the zone names as index
       self.adjacency_matrix: the adjacency matrix of the zones in the building
       self.fullsimulation: the full simulation df with the warm start
       self.simulation: the simulation df without the warm start and the useless variables
       self.simulationvariables: the whole list of the variables to collect
       self.useless_variables: a list of the useless variables (i.e., always 0).
        This list is computed when the simulation is set (method: set_simulation).
       self.nb_batch255: the number of batch of 255 variables to collect
           (because of the limitation of ReadVarsESO to 256
           variables per batch. One is Date/Time which is always collected).
       self.simulationfrequency: the hourly frequency of the simulation
       self.desiredfrequency: the desired frequency of the simulation (e.g., 1 for 1h, 4 for 15min, etc.)
       self.nb_warmupts: the number of time steps to use for sizing days simulation
       self.fullsimulation_expost: the full simulation df with the warm start for the ex-post simulation
       self.simulation_expost: the simulation df without the warm start and the useless variables for the ex-post simulation
       self.nb_warmupts_expost: the number of time steps to use for sizing days simulation for the ex-post simulation
       self.idf_filepath_expost: the path to the idf file for the ex-post simulation
       self.expost_cost: the cost of the ex-post simulation, cost for activating the HVAC following the temperature requirements
       self.nd_load: the non-dispatchable load of the building
       self.expected_results: a dataframe with the expected results of the optimization. I.e, the ems decisions.
       self.ml_model_doc: a string which contains the characteristics of the model. To be saved in the final excel
       """

    def __init__(self, idf_filepath, weather_path, output_folderpath, zone_floor_acu: pd.DataFrame, adjacency_matrix=None):
        """
        :param name: name of the building model
        :param idf_filepath: path to the idf file of the model
        :param weather_path: patht to the weather file
        :param output_folderpath: where to store the output files of the simulation
        :param zone_floor_acu: a dataframe with the zone names, the floor, and the ACU names (name == None if zone is not conditioned)
                """
        zone_floor_acu["is_conditioned"] = [acu is not None
                                            for acu in
                                            zone_floor_acu["acu"]]
        # filepaths
        self.idf_filepath = idf_filepath
        self.weather_path = weather_path
        self.output_folderpath = output_folderpath

        # building description
        self.zones_df = zone_floor_acu.rename(columns={"zone": "name"})
        self.zone_assets = pd.Series([Zone(z, f) for z, f in
                                      zip(self.zones_df['name'],
                                          self.zones_df['floor'])],
                                     index=self.zones_df['name'])
        self.adjacency_matrix = adjacency_matrix
        self.nd_load = None
        self.rc_model = None
        self.rc_model_data = None

        # Simulation variables
        self.fullsimulation = None
        self.simulation = None
        self.simulationvariables = None
        self.useless_variables = None
        self.simulationfrequency = None
        self.desiredfrequency = None
        self.nb_warmupts = None

        # ex-post simulation
        self.fullsimulation_expost = None
        self.simulation_expost = None
        self.nb_warmupts_expost = None
        self.idf_filepath_expost = None
        self.hvac_expost_cost = None
        self.performance_loss = None

    @property
    def name(self):
        """
        Returns the name of the building model
        :return: a string with the name of the building model
        """
        return self.idf_filepath.stem

    @property
    def zone_names(self):
        """
        Returns the names of the zones in the building model
        :return: a list of the zone names
        """
        return self.zones_df["name"].tolist()

    @property
    def conditioned_zones_df(self):
        """
               Returns the dataframe of the zones in the building model with
                       an HVAC system (i.e., without the plenums)
               :return: a dataframe with the zone names, the floor and the ACU
       names
                   without the plenums
               """
        return self.zones_df.loc[self.zones_df["is_conditioned"], :]

    @property
    def conditioned_zone_names(self):
        """
        Returns the names of the zones in the building model without the
            plenums
        :return: a list of the zone names without the plenums
        """
        return self.conditioned_zones_df["name"].tolist()

    @property
    def acus(self):
        """
        Returns the names of the ACUs in the building model and their floor
        :return: a list of the ACU names
        """
        return self.conditioned_zones_df.loc[
            :, ["acu", "floor"]].drop_duplicates()

    @property
    def nb_floors(self):
        """
        Returns the number of floors in the building model
        :return: an integer with the number of floors
        """
        return self.zones_df["floor"].max() + 1

    @property
    def nb_zones(self):
        """
        Returns the total number of zones in the building model,
        conditioned or not
        """
        return len(self.zone_names)

    @property
    def nb_conditioned_zones(self):
        """
        Returns the total number of conditioned zones in the building model
        """
        return len(self.conditioned_zone_names)

    @property
    def nb_batch255(self):
        """
        The number of batch of 255 variables to collect
        Returns
        -------

        """
        return len(self.simulationvariables) // 255

    @property
    def simulationvariables_batch255(self):
        """
        Divide the list of desired simulation variables
        (self.simulationvariables) into batches of 255 variables
        (because of the limitation of ReadVarsESO to 256
        Returns
        -------

        """
        return (
                [self.simulationvariables[i * 255:(i + 1) * 255] for i in
                 range(self.nb_batch255)]
                + [self.simulationvariables[self.nb_batch255 * 255:]])

    @property
    def simulation_exante(self):
        return self.simulation

    @simulation_exante.setter
    def simulation_exante(self, sim: pd.DataFrame):
        self.simulation = sim

    @property
    def temperature_mae(self):
        return sum([z.temperature_mae for z in self.zone_assets if
                    z.controlled]) / len(self.zone_assets)

    @property
    def temperature_max_e(self):
        return max(
            [z.temperature_max_e for z in self.zone_assets if
             z.controlled])

    @property
    def controlled_zone_names(self):
        """
        Return the names of the controlled zones. Should be the same as
        self.conditioned_zone_names.
        """
        return [z.name for z in self.zone_assets if z.controlled]

    @property
    def conditioned_zone_assets(self):
        """
        Return the list of the zone assets which are conditioned.
        This formulation is much more robust than:
            cza = [z for z in self.zone_assets if z.conditioned]
        because this formulation is valid as soon as the creation
        (via __init__) of the BuildingModel object. The other formulation
        is only valid after set_zone_hvac has been called.
        """
        cza = []
        for z in self.zone_assets:
            if z.name in self.conditioned_zone_names:
                cza.append(z)
        return cza

    @property
    def expost_results(self):
        dfs = [z.expost_results for z in self.conditioned_zone_assets]
        cols = [z.name for z in self.conditioned_zone_assets]
        return pd.concat(dfs, axis=1, keys=cols,
                         names=['zone', 'variable'])

    def set_simulationvariables(self, simulationvariables: list,
                                simulationfrequency: int,
                                desiredfrequency: int = 1):
        """
        Parameters
        ----------
        simulationvariables: list of the variables to collect (the whole
list)
        simulationfrequency: int indicating the frequency of the simulation
            (e.g., 4 for 15min, 1 for 1h)
        desiredfrequency: int indicating the desired frequency, i.e.,
            the frequency at which to resample the data

        """
        self.simulationvariables = simulationvariables
        self.simulationfrequency = simulationfrequency
        self.desiredfrequency = desiredfrequency
        self.nb_warmupts = None

    def set_nb_warmupts(self, n: int):
        self.nb_warmupts = n

    def set_simulations(self, fullsimulation: pd.DataFrame):
        """
        The input must be a full simulation with the warm start.
        The warm start is removed and the useless variables.
        :param fullsimulation:
        :return:
        """
        # Look for useless variables
        useless_variables = []
        for name, series in fullsimulation.items():
            if series.min() == series.max():
                useless_variables.append(name)
        self.useless_variables = useless_variables
        simvar = fullsimulation.columns
        self.simulationvariables = [v for v in simvar if
                                    v not in useless_variables]
        self.fullsimulation = fullsimulation.loc[:,
        self.simulationvariables]
        # nb_warmupts must have been defined beforehand
        self.simulation = self.fullsimulation.loc[
            ~self.fullsimulation["warmup"], :]

    def save_simulation(self, path, save_fullsimulation=False):
        """
        Save the full simulation in an excel and hdf file
        :param path: path is a path containing the name of the file
            without extension (e.g., "Model3/eplusout")
        :param save_fullsimulation: if True, save the full simulation
            (with the warm start) else, remov the warm_start
        :return:
        """
        df = self.fullsimulation if save_fullsimulation else self.simulation
        # memory efficient but long to save and load
        df.to_excel(path.with_suffix('.xlsx'), index=True, header=True, na_rep="NaN", freeze_panes=(1, 0))
        df.to_csv(path.with_suffix('.csv'), index=True, header=True,
                  na_rep="NaN")
        df.to_hdf(path.with_suffix('.h5'), index=True, key="/df", mode="w",
                  nan_rep="NaN")

    def load_simulation(self, filepath_hdf: str, tz: str = "UTC"):
        """
        load the simulation from the path
        :param path: path to the hdf file
        :return:
        """
        with pd.HDFStore(filepath_hdf, 'r') as store:
            self.simulation = pd.read_hdf(store, key="/df", mode="r")
        self.simulation.index = pd.DatetimeIndex(self.simulation.index,
tz=tz)

    def set_nondispatchable_load(self, load: pd.Series):
        """
        Set the non-dispatchable load of the building in kW
        :param load: pd.DataFrame with the non-dispatchable
        :return:
        """
        self.nd_load = load

    def set_expost(self):
        """
        Set the ex-post simulation for each zone in the building
        """
        for z in self.conditioned_zone_assets:
                cols = [f"Zone Mean Air Temperature,{z.name}",
                        f"Zone PACU Electricity Energy,{z.name}[Wh]"]
                # rows of interest are after warmup and over the scheduling
                # period
                mask1 = ~self.simulation_expost["warmup"]
                mask2 = (z.expected_results.index[0]
                         <= self.simulation_expost.index)
                mask3 = (self.simulation_expost.index
                         <= z.expected_results.index[-1])
                mask = mask1 & mask2 & mask3
                df = self.simulation_expost.where(mask).dropna()
                z.expost_results["t_in"] = df[cols[0]]
                z.expost_results["p_hvac"] = df[cols[1]].div(1000)

    def hvac_expost_cost(self, ts, market):
        """
        Compute the ex-post cost of the HVAC ONLY !!!

        :param ts: The DatetimeIndex of the ems decisions
            (must have a freq attribute)
        :param market: The prices of the electricity in €/kWh
        """
        # Output the import and export cost from EnergyPlus
        hvac_powers = [f"Zone PACU Electricity Energy,{z.name}[Wh]" for z in self.zone_assets if z.controlled]
        ts_sim = (self.simulation_expost.index >= ts[0]) & (
                self.simulation_expost.index < ts[-1])
        agg_hvac_power = self.simulation_expost.loc[ts_sim, hvac_powers].sum(axis=1).resample(ts.freq).sum()
        self.hvac_expost_cost = 0
        for t in ts[:-1]:
            price = market.prices_import.loc[t] if agg_hvac_power.loc[
                                                       t] >= 0 else \
                market.prices_export.loc[t]
            self.hvac_expost_cost += agg_hvac_power[t] * price / 1000
        self.hvac_expost_cost += max(0, *agg_hvac_power) * market.demand_charge

    def set_zone_hvac(self, T0: pd.Timestamp, Ttgt: pd.DataFrame):
        """
        Set the HVAC parameters (i.e., initial time, initial temperature,
        target temperature, minimum temperature, maximum temperature,
        the acu conditioning the zone, the acu capacity) for each zone
        in the building
        Parameters
        ----------
        T0: pd.Timestamp
            The initial time of the simulation
        Ttgt:
            The target temperatures for each zone. The columns are the zone
            names and the index is the time index (same as
simulation.index).

        Returns
        -------

        """
        for z in self.conditioned_zone_assets:
            z.set_HVAC(self, T0, Ttgt[z.name])

    def set_zone_occupancy_weights(self, yearly_occ: pd.DataFrame):
        """
        Set the occupancy weights for each zone in the building
        Parameters
        ----------
        yearly_occ: pd.DataFrame
            The occupancy weights for each zone. The columns are the zone
            names and the index is the time index (same as simulation.index).
        Returns
        -------
        """
        for z in self.zone_assets:
            z.set_occupancy_weights(yearly_occ[z.name])







class Zone:
    """
    Zone is a class which contains the necessary information about a zone
        in a building.
    self.name: str
        The name of the zone
    self.floor: int
        The floor of the zone
    self.controlled: bool
        Whether the zone is controlled or not.
        Is there an imposed temperature range?
    self.acu: str
        The name of the ACU which controls the zone
    self.Tin0: float
        The initial temperature of the zone
    self.Tmax: pd.Series
        The maximum temperature of the zone
    self.Tmin: pd.Series
        The minimum temperature of the zone
    self.Ttgt: pd.Series
        The target temperature of the zone for optimization
    self.hvac_capacity: pd.Series
        The electricity hvac capacity of the zone in kWe
    self.Tin_sim: pd.Series
        The simulated temperature of the zone
    self.simulation: pd.DataFrame
        The full simulation of the zone
    self.expected_results: pd.DataFrame
        The expected results of the optimization in a dataframe containing
        the ems decisions: the indoor temperature and HVAC power profiles
    self.expost_results: pd.DataFrame
        The ex-post results of the optimization in a dataframe containing
the
        indoor temperature and HVAC power profiles as returned by energy
plus
    self.occupancy_weights: pd.Series
        The occupancy weights of the building used to penalize
        the temperature deviation in the obj function
    """

    def __init__(self, name: str, floor: int):
        self.name = name
        self.floor = floor
        self.controlled = False
        self.acu = None
        self.Tin0 = None
        self.Tmax = None
        self.Tmin = None
        self.Ttgt = None
        self.hvac_capacity = None
        self.Tin_sim = None
        self.simulation = None
        self.expected_results = pd.DataFrame()
        self.expost_results = pd.DataFrame()
        self.occupancy_weights = None

    @property
    def conditioned(self):
        return self.controlled

    def set_HVAC(self, bldg: BuildingModel, initial_datetime: pd.Timestamp,
                 Ttgt: pd.Series):
        """
        Set the HVAC parameters of the zone, including the hvac capacity in
kW
        :param bldg:
        :param initial_datetime:
        :param Ttgt: pd.Series
            The target temperature profile for the zone (ideal temperature)
            for the whole simulation period (same index as bldg.simulation)
        :return:

        """
        if initial_datetime.tz != bldg.simulation.index.tz:
            raise ValueError(
                "The time zone of the initial_datetime and the simulation"
                "index must be the same.")
            # ensure the expected and expost results dataframes are empty
        self.expected_results = pd.DataFrame()
        self.expost_results = pd.DataFrame()
        # extract the row as a series thanks to squeeze (and not as a df)
        zone_row = bldg.zones_df.loc[
            bldg.zones_df["name"] == self.name].squeeze()
        self.controlled = zone_row["is_conditioned"]
        self.acu = zone_row["acu"]
        self.Tin0 = \
            bldg.simulation[f"Zone Mean Air Temperature,{self.name}"].loc[
                initial_datetime]
        if self.controlled:
            self.Tmax = bldg.simulation[
                f"Zone Thermostat Cooling Setpoint Temperature, {self.name}"]
            self.Tmin = bldg.simulation[
                f"Zone Thermostat Heating Setpoint Temperature, {self.name}"]
            # self.Tmax = pd.Series(index=bldg.simulation.index, data=29)
            # self.Tmin = pd.Series(index=bldg.simulation.index, data=16)
            freq = bldg.simulation.index.freq.nanos / 3.6e12  # in hours
            self.hvac_capacity = bldg.simulation[
                f"Zone PACU Electricity Energy, {self.name}[Wh]"].div(1000*freq).max()
            self.Ttgt = Ttgt
            # date = bldg.simulation[
            #     f"Zone PACU Electricity Energy,{self.acu}[Wh]"].idxmax()
            # print(f"Max HVAC capacity of {self.name} occurs on {date}")

    def set_occupancy_weights(self, weights: pd.Series):
        self.occupancy_weights = weights


    def set_simulation_results(self, simulation: pd.DataFrame):
        self.simulation = simulation
        self.Tin_sim = simulation[f"Zone Mean Air Temperature,{self.name}"]


    @property
    def temperature_mae(self):
        return (self.expected_results["Tin"] - self.expost_results[
            "Tin"]).abs().mean()


    @property
    def temperature_max_e(self):
        return (self.expected_results["Tin"] - self.expost_results[
            "Tin"]).abs().max()