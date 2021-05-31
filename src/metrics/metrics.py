from pathlib import Path
from src.metrics.utils import weighted_corr

import xarray as xr
import pandas as pd
import numpy as np


class ScoresTables:
    """
    Class to compute and save the tables with the scores:
    NMAE, BIAS, RMSE, PEARSON CORRELATION, Debiased-NMAE
    """
    def __init__(
            self,
            station_path_observations: Path,
            station_path_forecast: Path,
            metrics_output_dir: Path = Path('../../reports/tables/')
    ):
        self.observations = xr.open_dataset(station_path_observations,
                                            chunks={'time': 100})
        self.forecasts = xr.open_dataset(station_path_forecast,
                                         chunks={'time': 100})
        params = station_path_observations.name.split('_')
        self.location_parameters = {}
        for i, param in enumerate(['variable', 'country', 'city',
                                   'location_id', 'time_range']):
            self.location_parameters[param] = params[i]
        self.metrics_output_dir = metrics_output_dir

    def run(self):
        diff = self.forecasts - self.observations
        tables = [self.nmae_tables(diff, self.observations),
                  self.bias_tables(diff),
                  self.rmse_tables(diff),
                  self.nmae_debiased(self.forecasts, self.observations),
                  self.pearson_tables(self.forecasts, self.observations)]
        table_merged = self.merging_tables(tables)
        output_path = self.get_output_path()
        print("Writing table to {}".format(output_path))
        table_merged.to_excel(output_path,
                              float_format='%.2f')

    def nmae_tables(self, diff: xr.Dataset, obs: xr.Dataset):
        varname = self.location_parameters['variable']
        abs_diff = np.abs(diff)
        numerator = abs_diff.mean(dim="time")[varname]
        nmae = numerator / obs.mean(dime="time")
        return nmae

    def bias_tables(self, diff: xr.Dataset):
        varname = self.location_parameters['variable']
        bias = diff.mean(dim='time')[varname]
        return bias

    def nmae_debiased(self, forecast: xr.Dataset, observation: xr.Dataset):
        diff = forecast - observation
        bias = diff.mean(dim='time').mean(dim='study_case')
        debiased_forecast = forecast - bias
        diff = debiased_forecast - observation
        return self.nmae_tables(diff, observation)

    def rmse_tables(self, diff: xr.Dataset):
        varname = self.location_parameters['variable']
        rmse = np.sqrt((diff ** 2).mean(dim='time')[varname])
        return rmse

    def pearson_tables(self, forecast: xr.Dataset, observation: xr.Dataset):
        varname = self.location_parameters['variable']
        pearson_correlation = weighted_corr(forecast[varname],
                                            observation[varname],
                                            dim='time',
                                            weights=None,
                                            return_p=False)
        return pearson_correlation

    def merging_tables(self, tables):
        table_merged = pd.concat(tables, axis=1,
                                 keys=("NMAE", "Bias", "RMSE",
                                       'De-Biased NMAE', 'Pearson Correlation'))
        return table_merged

    def get_output_path(self):
        opath = Path(self.metrics_output_dir,
                     self.location_parameters['country'],
                     self.location_parameters['city'],
                     self.location_parameters['location_id'],
                     self.location_parameters['variable'],
                     f"{self.location_parameters['variable']}_"
                     f"{self.location_parameters['country']}_"
                     f"{self.location_parameters['city']}_"
                     f"{self.location_parameters['location_id']}_"
                     f"{self.location_parameters['time_range']}_metrics.xlsx")
        return opath